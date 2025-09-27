"""
Custom decorators for authentication, caching, and rate limiting
"""
from functools import wraps
from flask import jsonify
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
import redis
import json
import time
import hashlib

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            verify_jwt_in_request()
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': 'Authentication required', 'message': str(e)}), 401
    return decorated_function

def rate_limit(max_calls=100, time_window=3600):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                user_id = get_jwt_identity()
                key = f"rate_limit:{user_id}:{f.__name__}"
                
                # Get redis client from app context
                from flask import current_app
                redis_client = current_app.config.get('redis_client')
                
                if redis_client:
                    current = redis_client.get(key)
                    if current:
                        current = int(current)
                        if current >= max_calls:
                            return jsonify({'error': 'Rate limit exceeded'}), 429
                        redis_client.incr(key)
                    else:
                        redis_client.setex(key, time_window, 1)
                
                return f(*args, **kwargs)
            except Exception as e:
                # If rate limiting fails, allow the request
                return f(*args, **kwargs)
        return decorated_function
    return decorator

def cache_result(expiration=300):
    """Cache decorator for expensive operations"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{f.__name__}:{str(args)}:{str(kwargs)}"
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            
            try:
                from flask import current_app
                redis_client = current_app.config.get('redis_client')
                
                if redis_client:
                    # Try to get from cache
                    cached = redis_client.get(cache_key)
                    if cached:
                        return json.loads(cached)
                    
                    # Execute function
                    result = f(*args, **kwargs)
                    
                    # Cache the result
                    redis_client.setex(cache_key, expiration, json.dumps(result, default=str))
                    
                    return result
                else:
                    return f(*args, **kwargs)
            except Exception:
                return f(*args, **kwargs)
        return decorated_function
    return decorator
