# Importing
from flask import Flask, request, render_template, jsonify
import os
import base64
import google.generativeai as genai
from pymongo import MongoClient
from flask_cors import CORS, cross_origin
from PIL import Image
import io
import bcrypt
import pdfplumber
from sentence_transformers import SentenceTransformer, util

# Load BERT model
model_ats = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)
CORS(app) # Avoid Blocking


# Chatbot
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    try:
        if 'who trained' in user_message.lower() or 'who created' in user_message.lower() or 'who made' in user_message.lower():
            bot_response = "I was trained by the talented team ISS057 to assist you with your resume-building needs!"
        else:
            prompt = f"You are a resume-building assistant. Respond to this user query in plain text, without headings, quotations, or markdown formatting: {user_message}"
            response = model.generate_content(prompt)
            bot_response = response.text
    except Exception as e:
        bot_response = f"Error: {str(e)}"
    
    return jsonify({'response': bot_response})

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db_user = client['Login']
users_collection = db_user['users']

# Initialize default user if collection is empty
def init_default_user():
    if users_collection.count_documents({}) == 0:
        hashed_password = bcrypt.hashpw('User@123'.encode('utf-8'), bcrypt.gensalt())
        default_user = {
            'username': 'user',
            'password': hashed_password.decode('utf-8')
        }
        users_collection.insert_one(default_user)

init_default_user()

# Login Check
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = users_collection.find_one({'username': username})
    
    if not user:
        return jsonify({'success': False, 'error': 'username', 'message': 'User not found'}), 401
    
    if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return jsonify({'success': False, 'error': 'password', 'message': 'Incorrect password'}), 401
    
    return jsonify({'success': True, 'username': username})

# Register Storage
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if users_collection.find_one({'username': username}):
        return jsonify({'success': False, 'error': 'newUsername', 'message': 'Username already exists'}), 400

    # Password validation
    password_regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*])[A-Za-z\d!@#$%^&*]{8,}$'
    import re
    if not re.match(password_regex, password):
        return jsonify({
            'success': False,
            'error': 'newPassword',
            'message': 'Password must be at least 8 characters and contain one uppercase, one lowercase, one number, and one special character (!@#$%^&*)'
        }), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    new_user = {
        'username': username,
        'password': hashed_password.decode('utf-8')
    }
    
    users_collection.insert_one(new_user)
    return jsonify({'success': True})



# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["User"]
collection = db["users"]

os.environ["GEMINI_API_KEY"] = "AIzaSyDTvYF9fFaxGE_bvgaif-foRjdqK13P5e8" # Gemini API Key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# AI Model Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)


# Details Storing and Generate Resume
@app.route("/resume", methods=["POST"])
@cross_origin()
def generate_resume():
    # Extract user details
    full_name = request.json.get("name", "")
    email = request.json.get("email", "")
    phone_number = request.json.get("phone", "")
    linkedin = request.json.get("linkedin", "")
    city = request.json.get("city", "")
    state = request.json.get("state", "")
    country = request.json.get("country", "")
    role = request.json.get("role", "")
    skills = request.json.get("skills", [])
    certifications = request.json.get("certifications", [])
    projects = request.json.get("projects", [])
    work_experience = request.json.get("workExperience", [])
    education = request.json.get("education", [])
    internships = request.json.get("internships", [])
    languages = request.json.get("languages", [])
    template = request.json.get("template", "")

    # Process Image (Convert Base64 to JPEG)
    image = request.json.get("image", "")
    if image:
        image_data = image.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image_frame = Image.open(io.BytesIO(image_bytes))
        img_io = io.BytesIO()
        image_frame = image_frame.convert("RGB")
        image_frame.save(img_io, format="JPEG")
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")
        jpeg_image = f"data:image/jpeg;base64,{img_base64}"
    else:
        jpeg_image = ""

    # Generate About Me Section using Gemini AI
    about_me_prompt = f"""
    Write a concise and ATS-friendly professional summary in 3-5 lines for {full_name}, a {role}. Incorporate key skills: {', '.join(skills)} naturally within the text.Include Self Intro like "I am .."  at first .Avoid headings, quotations, and bullet points.
    """
    response = model.start_chat().send_message(about_me_prompt)
    about_me = response.text

    # Generate Project, Experience, Internship Summaries
    generated_project_summaries = [
    model.start_chat().send_message(
        f"Write a 3-4 line professional summary from my prespective(I am) for the project '{p['title']}' which involved '{p['technologies']}' with '{p['description']}'. Describe its purpose, key contributions and impact concisely without bullet points or heading ."
        ).text
        for p in projects
    ]

    generated_experience_summaries = [
    model.start_chat().send_message(
        f"Write a 3-4 line professional summary for the role '{w['title']}' at '{w['company']}' with '{w['responsibilities']}' . "
        f"Summarize key responsibilities, contributions, and impact in a clear and concise way from my prespective(I am)"
        f"without bullet points or headings."
        ).text
        for w in work_experience
    ]

    generated_internship_summaries = [
    model.start_chat().send_message(
        f"Write a 3-4 line summary for the internship role '{i['role']}' at '{i['company']}'  with '{i['responsibilities']}' . "
        f"Highlight key responsibilities, contributions, and impact in a professional and concise mannerfrom my prespective(I am)"
        f"without bullet points or headings."
        ).text
        for i in internships
    ]

    # Combine data
    combined_experiences = [{'work': e, 'summary': s} for e, s in zip(work_experience, generated_experience_summaries)]
    combined_projects = [{'project': p, 'summary': s} for p, s in zip(projects, generated_project_summaries)]
    combined_internships = [{'internship': i, 'summary': s} for i, s in zip(internships, generated_internship_summaries)]

    # Create a dictionary to store all user details
    user_data = {
        "full_name": full_name,
        "email": email,
        "phone_number": phone_number,
        "linkedin": linkedin,
        "city": city,
        "state": state,
        "country": country,
        "role": role,
        "skills": skills,
        "certifications": certifications,
        "projects": combined_projects,
        "work_experience": combined_experiences,
        "education": education,
        "internships": combined_internships,
        "languages": languages,
        "image": jpeg_image,
        "about_me": about_me,
        "template": template
    }

    # Insert the data into MongoDB
    collection.insert_one(user_data)
    print("Data Stored Successfully")
    # Print the data
    print(f"Full Name: {full_name}")
    print(f"Email: {email}")
    print(f"Phone: {phone_number}")
    print(f"LinkedIn: {linkedin}")
    print(f"City: {city}")
    print(f"State: {state}")
    print(f"Country: {country}")
    print(f"Role: {role}")
    print(f"Skills: {skills}")
    print(f"Certifications: {certifications}")
    print(f"Projects: {projects}")
    print(f"Work Experience: {work_experience}")
    print(f"Education: {education}")
    print(f"Internships: {internships}")
    print(f"Languages: {languages}")
    print(f"Image (Base64): {jpeg_image}")
    print(f"Combined Experiences: {combined_experiences}")
    print(f"Combined Projects: {combined_projects}")
    print(f"Combined Internships: {combined_internships}")
    print(f"Template: {template}")

    # Render HTML Resume
    if template=="resume1":
        return render_template(
            "Resume_Template1.html",
            full_name=full_name,
            email=email,
            phone_number=phone_number,
            linkedin=linkedin,
            city=city,
            state=state,
            country=country,
            role=role,
            about_me=about_me,
            skills=skills,
            certifications=certifications,
            combined_projects=combined_projects,
            combined_experiences=combined_experiences,
            education=education,
            combined_internships=combined_internships,
            languages=languages,
            image=jpeg_image
        )
    
    if template=="resume2":
        return render_template(
            "Resume_Template2.html",
            full_name=full_name,
            email=email,
            phone_number=phone_number,
            linkedin=linkedin,
            city=city,
            state=state,
            country=country,
            role=role,
            about_me=about_me,
            skills=skills,
            certifications=certifications,
            combined_projects=combined_projects,
            combined_experiences=combined_experiences,
            education=education,
            combined_internships=combined_internships,
            languages=languages,
            image=jpeg_image
        )
    
    if template=="resume3":
        return render_template(
            "Resume_Template3.html",
            full_name=full_name,
            email=email,
            phone_number=phone_number,
            linkedin=linkedin,
            city=city,
            state=state,
            country=country,
            role=role,
            about_me=about_me,
            skills=skills,
            certifications=certifications,
            combined_projects=combined_projects,
            combined_experiences=combined_experiences,
            education=education,
            combined_internships=combined_internships,
            languages=languages,
            image=jpeg_image
        )


# ATS
def extract_text_from_pdf(file):
    """Extract text from a PDF file object in memory."""
    text = ""
    try:
        # Use io.BytesIO to treat the file content as a file-like object in memory
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""
    return text.strip()

def generate_feedback(resume_text, job_description, model_score):
    """Generate friendly feedback based on resume, job description, and model score."""
    job_keywords = set(job_description.lower().split())
    resume_keywords = set(resume_text.lower().split())
    
    # Calculate key metrics
    missing_keywords = job_keywords - resume_keywords
    matching_keywords = job_keywords & resume_keywords
    match_percentage = (len(matching_keywords) / len(job_keywords)) * 100 if job_keywords else 0
    resume_word_count = len(resume_text.split())
    score_percentage = model_score * 100  # Convert model score (0-1) to percentage
    
    # Initialize feedback list
    feedback_parts = []

    # Feedback based on keyword match percentage
    if match_percentage >= 80:
        feedback_parts.append("Awesome! Your resume looks like a great fit for this job based on keywords.")
    elif match_percentage >= 50:
        feedback_parts.append("Nice work! Your resume is on the right track with keywords, but we can make it even better.")
    elif match_percentage >= 20:
        feedback_parts.append("You’ve got a start with keywords, but your resume could use some extra love to match this job.")
    else:
        feedback_parts.append("It looks like your resume might need more tailoring to match the job’s keywords.")

    # Feedback on missing keywords
    if missing_keywords:
        feedback_parts.append(f"Try sprinkling in these words to boost your fit: {', '.join(sorted(missing_keywords))}.")
    else:
        feedback_parts.append("You’ve nailed it—no important words are missing from the job description!")

    # Feedback on matching keywords
    if matching_keywords:
        feedback_parts.append(f"You’re already rocking words like {', '.join(sorted(list(matching_keywords)[:3]))}—keep it up!")
    else:
        feedback_parts.append("We couldn’t find any key words from the job in your resume yet.")

    # Feedback based on model score
    if score_percentage >= 80:
        feedback_parts.append("Fantastic! The deeper analysis shows your resume is a super strong match for this role.")
    elif score_percentage >= 60:
        feedback_parts.append("Good news! The analysis suggests your resume is a solid fit, with a little room to shine brighter.")
    elif score_percentage >= 40:
        feedback_parts.append("Not bad! The analysis sees some connection to the job, but tweaking it could lift your score.")
    else:
        feedback_parts.append("Heads up! The deeper analysis thinks your resume could use more work to align with this job.")

    # Feedback based on resume length
    if resume_word_count < 100:
        feedback_parts.append("Your resume’s a little short. Adding more details could help it stand out!")
    elif resume_word_count > 500:
        feedback_parts.append("Your resume’s pretty long. Shortening it a bit might make it easier to read.")
    else:
        feedback_parts.append("The length of your resume feels just right—nice balance!")

    # Combine feedback into a single string
    feedback = " ".join(feedback_parts)
    return feedback

def rank_resumes(resume_files, job_description):
    """Rank resumes based on similarity to the job description without saving to disk."""
    job_embedding = model_ats.encode(job_description, convert_to_tensor=True)
    resume_scores = {}
    resume_feedback = {}

    for file in resume_files:
        resume_text = extract_text_from_pdf(file)
        if not resume_text:
            print(f"Skipping {file.filename} (No text extracted)")
            continue

        resume_embedding = model_ats.encode(resume_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(job_embedding, resume_embedding)
        score = similarity.item()

        print(f"Resume: {file.filename} | Score: {score}")

        resume_scores[file.filename] = score
        resume_feedback[file.filename] = generate_feedback(resume_text, job_description, score)  # Pass score here

    ranked_resumes = sorted(resume_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_resumes, resume_feedback

@app.route("/upload/", methods=["POST"])
def upload_files():
    """Handle file uploads and return ranking results without saving to disk."""
    if "job_description" not in request.form:
        return jsonify({"message": "Job description is required"}), 400

    job_description = request.form["job_description"]
    if "resumes" not in request.files:
        return jsonify({"message": "No resumes uploaded"}), 400

    resumes = request.files.getlist("resumes")
    if not resumes or all(file.filename == "" for file in resumes):
        return jsonify({"message": "No resumes uploaded"}), 400

    for file in resumes:
        if not file or not file.filename.endswith(".pdf"):
            return jsonify({"message": "Please upload only PDF files"}), 400

    print(f"Received files: {[file.filename for file in resumes]}")

    ranked_results, feedback_results = rank_resumes(resumes, job_description)

    if not ranked_results:
        return jsonify({"message": "No resumes analyzed. Please try again!"}), 400

    response = [{
        "resume": resume,
        "score": round(score * 100, 2),
        "feedback": feedback_results[resume]
    } for resume, score in ranked_results]

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)