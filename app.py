import pandas as pd
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import json
import firebase_admin
from firebase_admin import credentials, auth, db
from flask import session, redirect, url_for, flash
import requests
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify 
import gdown

# --- Data File Downloader ---
# List of files your app needs
REQUIRED_FILES = [
    'recipes.csv',
    'recipes_ingredients.csv',
    'recipes_master_list.csv',
    'tfidf_matrix.npz',
    'tfidf_vectorizer.pkl'
]

# Check if all files exist locally
all_files_exist = all(os.path.exists(f) for f in REQUIRED_FILES)

if not all_files_exist:
    print("INFO: Data files not found. Downloading from Google Drive...")

    gdrive_url = "https://drive.google.com/drive/folders/1RAIrTJobyoTomYZ-Wjb_L_qsAM2pKlHl?usp=sharing" 
   

    try:
        # Download all files from the folder
        gdown.download_folder(gdrive_url, quiet=False, use_cookies=False)
        print("INFO: Download complete.")
    except Exception as e:
        print(f"FATAL: Failed to download data files: {e}")
else:
    print("INFO: Data files already exist. Skipping download.")

# --- End of Downloader ---

print("--- Starting Server ---")
print("Loading all assets... This might take a moment.")


# Get the full path to the directory where this app.py file lives
APP_PATH = os.path.dirname(os.path.abspath(__file__))

try:
    # Build the full path to each model file
    df_path = os.path.join(APP_PATH, "recipes_master_list.csv")
    vectorizer_path = os.path.join(APP_PATH, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(APP_PATH, "tfidf_matrix.npz")
    
    # Load the "details" CSV
    df = pd.read_csv(df_path)
    
    # Load (the TF-IDF vectorizer)
    with open(vectorizer_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
        
    # Load  (the TF-IDF matrix)
    tfidf_matrix = load_npz(matrix_path)

except FileNotFoundError as e:
    print("---")
    print(f"FATAL ERROR: A model file was not found: {e}")
    print("Please make sure all 3 model files are in the same folder as app.py.")
    print("---")
    exit()

# --- FIREBASE INITIALIZATION ---
try:
    # 1. Try to get the credentials from the server's Environment Variable
    cred_json_str = os.environ.get('FIREBASE_CREDENTIALS')

    if cred_json_str:
        # If running on the server (Render/Heroku), load from the text
        print("INFO: Loading Firebase credentials from Environment Variable.")
        cred_dict = json.loads(cred_json_str)
        cred = credentials.Certificate(cred_dict)
    else:
        # If running locally, fall back to your local key file
        print("WARNING: FIREBASE_CREDENTIALS not set. Falling back to local file.")
        # ↓↓↓ IMPORTANT: PUT YOUR NEW KEY'S FILENAME HERE ↓↓↓
        local_key_path = "matbahakk-firebase-adminsdk-fbsvc-edca83a4c1.json" 
        cred = credentials.Certificate(local_key_path)

    # Initialize the app (this line stays the same)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://matbahakk-default-rtdb.firebaseio.com/'
    })

    # Your Web API Key from the Firebase "General" settings
    FIREBASE_WEB_API_KEY = "AIzaSyCIPL7kzhXCu3YUUCzLpMAAof3N0CMAUJE"
    
except ValueError:
    pass 
# --- END FIREBASE INITIALIZATION ---

# --- 2. INITIALIZE THE FLASK APP ---
app = Flask(__name__)

#handle user login sessions
app.config['SECRET_KEY'] = 'a-very-secret-and-random-key-123'

print("--- ASSETS LOADED. APP IS READY! ---")

# --- 4. THE RECOMMENDATION ENGINE ---
# --- 4. THE RECOMMENDATION ENGINE (UPDATED) ---
# --- 4. THE RECOMMENDATION ENGINE (UPDATED) ---
# --- 4. THE RECOMMENDATION ENGINE (FINAL VERSION) ---
def get_recommendations(user_ingredients, user_time, user_difficulty, user_rating):
    
    # --- 1. START WITH ALL RECIPES ---
    filtered_df = df.copy()
    
    # --- 2. APPLY FILTERS ---
    
    # Filter by Time
    if user_time != 'any':
        user_time_int = int(user_time)
        filtered_df = filtered_df[filtered_df['TotalTime'] <= user_time_int]
        
    # Filter by Difficulty
    if user_difficulty != 'any':
        filtered_df = filtered_df[filtered_df['Difficulty'] == user_difficulty]
        
    # Filter by Rating
    if user_rating != 'any':
        user_rating_float = float(user_rating)
        filtered_df = filtered_df[filtered_df['rating'].fillna(0) >= user_rating_float]
    
    if filtered_df.empty:
        return []
        
    # --- 3. GET THE MATRIX FOR FILTERED RECIPES ---
    filtered_indices = filtered_df.index
    filtered_tfidf_matrix = tfidf_matrix[filtered_indices]

    # --- 4. RUN THE ML MODEL (COSINE SIMILARITY) ---
    user_tfidf_vector = tfidf_vectorizer.transform([user_ingredients])
    cos_similarities = cosine_similarity(user_tfidf_vector, filtered_tfidf_matrix).flatten()
    
    # --- 5. RANK AND SORT ALL RESULTS (NEW LOGIC) ---
    
    # Assign scores by creating a new DataFrame from the similarities,
    # ensuring the index matches filtered_df's index.
    sim_df = pd.DataFrame(cos_similarities, index=filtered_indices, columns=['similarity'])
    
    # Now merge the scores back into filtered_df
    filtered_df = filtered_df.join(sim_df)

    # Only keep recipes with a good ingredient match score (greater than 0.4)
    filtered_df = filtered_df[filtered_df['similarity'] > 0.4]
    
    # Handle missing ratings (NaN) by filling with 0
    filtered_df['rating'] = filtered_df['rating'].fillna(0)

    # Sort by similarity FIRST, then rating
    sorted_df = filtered_df.sort_values(
        by=['similarity', 'rating' ], 
        ascending=[False, False]
    )
    
    # --- 6. GET TOP 10 (AND THE ID) ---
    top_10_recipes = sorted_df.head(10).reset_index().rename(columns={'index': 'id'})
    
    # Convert to a list of dictionaries and return
    return top_10_recipes.to_dict('records')


# --- 3. WEBSITE ROUTES ---
@app.route('/')
def home():
    # This tells Flask to find "index.html" in the "templates" folder and show it.
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    recipes = []
    
    # --- HANDLE POST REQUEST (Initial Search Form Submission) ---
    if request.method == 'POST':
        print(f"FORM DATA RECEIVED: {request.form}")
        
        # 1. GET THE DATA FROM THE FORM
        user_ingredients = request.form.get('ingredients')
        user_time = request.form.get('TotalTime')
        user_difficulty = request.form.get('Difficulty')
        user_rating = request.form.get('rating')

        # 2. Save search criteria to session for later GET requests (like after a favorite save)
        session['last_search'] = {
            'ingredients': user_ingredients,
            'TotalTime': user_time,
            'Difficulty': user_difficulty,
            'rating': user_rating
        }
        
        # 3. CALL YOUR tf-idf FUNCTION
        recipes = get_recommendations(user_ingredients, user_time, user_difficulty, user_rating)
        
    # --- HANDLE GET REQUEST (Redirect from /add_favorite) ---
    elif request.method == 'GET':
        if 'last_search' in session:
            # Re-run search based on saved criteria to keep user on the results page
            last_search = session['last_search']
            
            recipes = get_recommendations(
                last_search['ingredients'], 
                last_search['TotalTime'], 
                last_search['Difficulty'], 
                last_search['rating']
            )
        else:
            # If no search was ever performed, redirect to home
            return redirect(url_for('home'))

    # --- FINAL STEP ---
    # Pass the recipes (from POST or GET) to results.html
    return render_template('results.html', recipes=recipes)

   # --- NEW ROUTE - LOOKS UP BY NAME ---
@app.route('/recipe/<recipe_name>')
def recipe_details(recipe_name):
    try:
        # 1. Find the recipe by its unique name instead of its ID
        # We use .iloc[0] because find returns a DataFrame, and we want the first (only) row
        recipe = df[df['name'] == recipe_name].iloc[0].to_dict()
        
        # 2. Convert the steps from a string "step 1 | step 2" into a real list
        if pd.isna(recipe['steps']):
            steps_list = ["No steps provided."]
        else:
            steps_list = recipe['steps'].split('|')
            steps_list = [step.strip() for step in steps_list]

        # 3. Pass the correct recipe
        return render_template('recipe_details.html', recipe=recipe, steps=steps_list)
    
    except IndexError:
        # This will catch if no recipe with that name is found
        return "Recipe not found!", 404

# ---AUTHENTICATION ROUTES---

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # --- 1. Get all three fields from the form ---
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        try:
            # --- 2. Create the user in Firebase Auth ---
            user = auth.create_user(
                email=email,
                password=password,
                display_name=name  
            )
            
            # --- 3. Save the name to our Realtime Database ---
            # We create a new 'users' folder and save the name using the user's ID
            db.reference(f'users/{user.uid}').set({
                'name': name,
                'email': email
            })
            
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            # Handle errors (like email already in use)
            flash(f'Error creating account: {e}', 'danger')
            return redirect(url_for('register'))
            
    # If it's a 'GET' request, just show the register page
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        try:
            # --- 1. Log the user in with email and password ---
            login_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
            data = {"email": email, "password": password, "returnSecureToken": True}
            response = requests.post(login_url, json=data)
            response.raise_for_status() # Check for errors
            
            user_data = response.json()
            user_id = user_data['localId'] # This is the user's unique ID
            
            # --- 2. Fetch the user's name from the Realtime Database ---
            user_info = db.reference(f'users/{user_id}').get()
            
            # --- 3. Save everything to the session ---
            session['user'] = user_id
            session['email'] = user_data['email']
            # Save the name (if it exists)
            session['name'] = user_info.get('name', 'User') if user_info else 'User'
            
            return redirect(url_for('home'))
            
        except requests.exceptions.HTTPError:
            flash('Login failed. Please check your email and password.', 'danger')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'An error occurred: {e}', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    # Clear the user from the session
    session.pop('user', None)
    session.pop('email', None)
    
    # Redirect to the homepage
    return redirect(url_for('home'))

@app.route('/add_favorite', methods=['POST'])
def add_favorite():
    # 1. Check if the user is logged in
    if 'user' not in session:
        # Redirect to login with NO flash message (as requested)
        return redirect(url_for('login'))
        
    recipe_name = request.form.get('recipe_name')
    user_id = session['user']
    
    if not recipe_name:
        # If recipe name is missing, redirect back quietly
        return redirect(request.referrer or url_for('home'))

    # 2. Define the path and fetch current favorites
    favorites_ref = db.reference(f'users/{user_id}/favorites')
    current_favorites = favorites_ref.get()
    
    # 3. Initialize/Normalize the favorites list
    # If Firebase returns None (no favorites yet), or something that isn't a list, 
    # start with an empty list.
    if not isinstance(current_favorites, list):
         current_favorites = [] 
        
    # 4. Check for duplicate before saving 
    if recipe_name not in current_favorites:
        # Add and save only if it's not a duplicate
        current_favorites.append(recipe_name)
        favorites_ref.set(current_favorites)
        
    # 5. Redirect back to the page you came from (the search results page)
    # This prevents the 405 crash because your /search route is now stable.
    return redirect(request.referrer or url_for('home'))

@app.route('/favorites')
def favorites():
    # Requires user to be logged in
    if 'user' not in session:
        return redirect(url_for('login'))
        
    user_id = session['user']
    
    # Fetch the list of favorite recipe names from Firebase
    favorites_ref = db.reference(f'users/{user_id}/favorites')
    favorite_names = favorites_ref.get()
    
    # Process data for the template
    if not favorite_names:
        recipes_to_display = []
    else:
        # Convert list of names to list of dictionaries for the template
        recipes_to_display = [{'name': name} for name in favorite_names]

    return render_template('favorites.html', recipes=recipes_to_display)

if __name__ == "__main__":
    app.run(debug=False)