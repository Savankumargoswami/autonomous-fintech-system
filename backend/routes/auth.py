"""
Authentication routes
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('auth', __name__)

@bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get user profile"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        db = current_app.config.get('db')
        
        if db:
            user = db.users.find_one({'_id': ObjectId(user_id)})
            if user:
                return jsonify({
                    'id': str(user['_id']),
                    'username': user['username'],
                    'email': user['email'],
                    'created_at': user['created_at'].isoformat()
                }), 200
        
        return jsonify({'error': 'User not found'}), 404
        
    except Exception as e:
        logger.error(f"Profile fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """Change user password"""
    try:
        from flask import current_app
        user_id = get_jwt_identity()
        data = request.get_json()
        
        old_password = data.get('old_password')
        new_password = data.get('new_password')
        
        if not all([old_password, new_password]):
            return jsonify({'error': 'Both old and new passwords required'}), 400
        
        db = current_app.config.get('db')
        if db:
            user = db.users.find_one({'_id': ObjectId(user_id)})
            
            if user and check_password_hash(user['password'], old_password):
                hashed_password = generate_password_hash(new_password)
                db.users.update_one(
                    {'_id': ObjectId(user_id)},
                    {'$set': {'password': hashed_password}}
                )
                return jsonify({'message': 'Password updated successfully'}), 200
            else:
                return jsonify({'error': 'Invalid old password'}), 401
        
        return jsonify({'error': 'Database error'}), 500
        
    except Exception as e:
        logger.error(f"Password change error: {e}")
        return jsonify({'error': str(e)}), 500
