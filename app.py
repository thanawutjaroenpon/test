#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web UI for Smoke Detector Auto-Placer
Simple Flask application for easy DXF file processing
"""

import os
import tempfile
import uuid
from pathlib import Path
from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import subprocess
import sys

app = Flask(__name__)
app.secret_key = 'smoke_detector_placer_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'dxf'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_{filename}"
        
        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get processing options
        standard = request.form.get('standard', 'NFPA72')
        grid_type = request.form.get('grid_type', 'square')
        spacing = request.form.get('spacing', '9.1')
        include_coverage = request.form.get('coverage', 'off') == 'on'
        coverage_radius = request.form.get('coverage_radius', '').strip()
        
        try:
            # Process the file (auto-detect offset)
            result = process_dxf_file(
                filepath,
                standard,
                grid_type,
                spacing,
                include_coverage=include_coverage,
                coverage_radius=coverage_radius
            )
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'message': result['message'],
                    'download_url': url_for('download_file', filename=result['output_filename']),
                    'output_filename': result['output_filename']
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result['message']
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error processing file: {str(e)}'
            })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid file type. Please upload a DXF file.'
        })

def process_dxf_file(filepath, standard='NFPA72', grid_type='square', spacing='9.1',
                     include_coverage=False, coverage_radius=''):
    """Process DXF file using the smoke detector placer"""
    try:
        # Generate output filename
        input_filename = Path(filepath).name
        output_filename = f"processed_{input_filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Build command
        cmd = [
            sys.executable, 
            'smoke_detector_placer.py',
            filepath,
            '--out', output_path,
            '--no-pdf'  # Skip PDF generation for web UI
        ]
        
        # Add standard if not default
        if standard != 'NFPA72':
            cmd.extend(['--std', standard])
        
        # Add grid type if not default
        if grid_type != 'square':
            cmd.extend(['--grid', grid_type])
        
        # USER REQUEST: ล็อค spacing ที่ 9.1 เมตรเสมอ
        # Always use 9.1 meters spacing (locked)
        cmd.extend(['--spacing', '9.1'])
        
        coverage_radius_value = None
        if coverage_radius:
            try:
                coverage_radius_value = float(coverage_radius)
            except ValueError:
                coverage_radius_value = None
        
        if include_coverage:
            cmd.append('--coverage-circles')
            if coverage_radius_value and coverage_radius_value > 0:
                cmd.extend(['--coverage-radius', str(coverage_radius_value)])
        elif coverage_radius_value:
            # Warn user in logs if radius provided without enabling coverage
            print("⚠️  Coverage radius provided but coverage circles disabled; ignoring.")
        
        # Run the command (offset will be auto-detected)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            # Check if output file was created
            if os.path.exists(output_path):
                # Parse output for statistics
                output_lines = result.stdout.split('\n')
                stats = {
                    'rooms': 'N/A',
                    'detectors': 'N/A',
                    'cleaned': 'N/A'
                }
                
                for line in output_lines:
                    if 'Found' in line and 'rooms' in line:
                        stats['rooms'] = line.split('Found')[1].split('rooms')[0].strip()
                    elif 'Placed' in line and 'detectors' in line:
                        stats['detectors'] = line.split('Placed')[1].split('detectors')[0].strip()
                    elif 'Cleaned' in line and 'detectors' in line:
                        stats['cleaned'] = line.split('Cleaned')[1].split('detectors')[0].strip()
                
                message = f"✅ Successfully processed! Found {stats['rooms']} rooms, placed {stats['detectors']} detectors"
                if stats['cleaned'] != 'N/A':
                    message += f", cleaned {stats['cleaned']} old detectors"
                
                return {
                    'success': True,
                    'message': message,
                    'output_filename': output_filename,
                    'stats': stats
                }
            else:
                return {
                    'success': False,
                    'message': 'Processing completed but output file not found'
                }
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return {
                'success': False,
                'message': f'Processing failed: {error_msg}'
            }
            
    except Exception as e:
        return {
            'success': False,
            'message': f'Error: {str(e)}'
        }

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed file"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            flash('File not found')
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Download error: {str(e)}')
        return redirect(url_for('index'))

@app.route('/inspect', methods=['POST'])
def inspect_file():
    """Inspect DXF file without processing"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        # Save file temporarily
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Run inspection
            cmd = [sys.executable, 'smoke_detector_placer.py', filepath, '--inspect']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            # Clean up temp file
            os.remove(filepath)
            
            if result.returncode == 0:
                return jsonify({
                    'success': True,
                    'inspection': result.stdout
                })
            else:
                return jsonify({
                    'success': False,
                    'message': result.stderr if result.stderr else 'Inspection failed'
                })
        except Exception as e:
            # Clean up temp file
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'success': False,
                'message': f'Inspection error: {str(e)}'
            })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid file type. Please upload a DXF file.'
        })

if __name__ == '__main__':
    print("🔥 Smoke Detector Auto-Placer Web UI")
    print("=" * 50)
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:8080")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=8080)
