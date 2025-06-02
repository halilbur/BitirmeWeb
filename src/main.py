from flask import Flask, render_template, request, jsonify
from datetime import datetime

# Create Flask application instance
app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('index.html', 
                         title='Flask Example',
                         current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# About page route
@app.route('/about')
def about():
    return render_template('about.html', title='About Us')

# Contact form route (GET and POST)
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # In a real app, you'd save this to a database
        return render_template('contact.html', 
                             title='Contact',
                             success=True,
                             name=name)
    
    return render_template('contact.html', title='Contact')

# API endpoint example
@app.route('/api/users')
def api_users():
    users = [
        {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'},
        {'id': 2, 'name': 'Bob', 'email': 'bob@example.com'},
        {'id': 3, 'name': 'Charlie', 'email': 'charlie@example.com'}
    ]
    return jsonify(users)

# Dynamic route with parameter
@app.route('/user/<int:user_id>')
def user_profile(user_id):
    # Mock user data
    users = {
        1: {'name': 'Alice', 'email': 'alice@example.com', 'joined': '2023-01-15'},
        2: {'name': 'Bob', 'email': 'bob@example.com', 'joined': '2023-02-20'},
        3: {'name': 'Charlie', 'email': 'charlie@example.com', 'joined': '2023-03-10'}
    }
    
    user = users.get(user_id)
    if user:
        return render_template('user_profile.html', 
                             title=f"Profile - {user['name']}", 
                             user=user, 
                             user_id=user_id)
    else:
        return render_template('404.html'), 404

# Error handler
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)