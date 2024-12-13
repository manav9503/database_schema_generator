from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# File paths for saving models and embeddings
model_path = 'logistic_model.pkl'
label_encoder_path = 'label_encoder.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

# Training data
data = [
 {"prompt": "A ticket booking system", "schema": "{\"Ticket\": {\"TicketID\": \"Number\", \"Event\": \"Lookup\", \"CustomerName\": \"Text\", \"SeatNumber\": \"Text\", \"Price\": \"Currency\"}}"},
    {"prompt": "An expense tracking app", "schema": "{\"Expense\": {\"Description\": \"Text\", \"Amount\": \"Currency\", \"Category\": \"Picklist\", \"Date\": \"Date\", \"PaidBy\": \"Lookup\"}}"},
    {"prompt": "A rental management system", "schema": "{\"Rental\": {\"ItemName\": \"Text\", \"Renter\": \"Lookup\", \"StartDate\": \"Date\", \"EndDate\": \"Date\", \"RentalFee\": \"Currency\"}}"},
    {"prompt": "A blogging platform", "schema": "{\"BlogPost\": {\"Title\": \"Text\", \"Author\": \"Lookup\", \"Content\": \"Rich Textarea\", \"PublishedDate\": \"Date\", \"Tags\": \"Picklist\"}}"},
    {"prompt": "A warehouse inventory system", "schema": "{\"Inventory\": {\"ItemName\": \"Text\", \"SKU\": \"Text\", \"Quantity\": \"Number\", \"Warehouse\": \"Lookup\", \"ReorderLevel\": \"Number\"}}"},
    {"prompt": "A CRM for tracking leads", "schema": "{\"Lead\": {\"Name\": \"Text\", \"Email\": \"Email\", \"Phone\": \"Phone\", \"Source\": \"Picklist\", \"Status\": \"Picklist\"}}"},
    {"prompt": "A student management system", "schema": "{\"Student\": {\"Name\": \"Text\", \"Email\": \"Email\", \"RollNumber\": \"Number\", \"DOB\": \"Date\", \"Grade\": \"Picklist\"}}"},
    {"prompt": "An app for managing employee records", "schema": "{\"Employee\": {\"Name\": \"Text\", \"Email\": \"Email\", \"Phone\": \"Phone\", \"JoiningDate\": \"Date\", \"Salary\": \"Currency\"}}"},
    {"prompt": "A simple CRM tool", "schema": "{\"Customer\": {\"CustomerName\": \"Text\", \"Email\": \"Email\", \"Phone\": \"Phone\", \"Address\": \"Textarea\"}}"},
    {"prompt": "A school management system", "schema": "{\"Student\": {\"Name\": \"Text\", \"Email\": \"Email\", \"RollNumber\": \"Number\", \"DOB\": \"Date\", \"Grade\": \"Picklist\"}}"},
    {"prompt": "An inventory management app", "schema": "{\"Product\": {\"ProductName\": \"Text\", \"SKU\": \"Text\", \"Price\": \"Currency\", \"Stock\": \"Number\", \"Category\": \"Picklist\"}}"},
    {"prompt": "A library management system", "schema": "{\"Book\": {\"Title\": \"Text\", \"Author\": \"Text\", \"ISBN\": \"Text\", \"PublishedYear\": \"Number\", \"Genre\": \"Picklist\"}}"},
    {"prompt": "A fitness tracking app", "schema": "{\"User\": {\"Name\": \"Text\", \"Age\": \"Number\", \"Email\": \"Email\", \"WorkoutHistory\": \"Rich Textarea\"}}"},
    {"prompt": "A food delivery app", "schema": "{\"Order\": {\"OrderID\": \"Number\", \"DeliveryAddress\": \"Textarea\", \"TotalAmount\": \"Currency\"}}"},
    {"prompt": "A healthcare management app", "schema": "{\"Patient\": {\"Name\": \"Text\", \"DOB\": \"Date\", \"MedicalHistory\": \"Rich Textarea\", \"Doctor\": \"Lookup\"}}"},
    {"prompt": "An event management app", "schema": "{\"Event\": {\"EventName\": \"Text\", \"Date\": \"Date\", \"Venue\": \"Text\", \"Organizer\": \"Lookup\"}}"},
    {"prompt": "A real estate management app", "schema": "{\"Property\": {\"Address\": \"Textarea\", \"Price\": \"Currency\", \"Owner\": \"Lookup\", \"Type\": \"Picklist\"}}"},
    {"prompt": "A travel booking system", "schema": "{\"Booking\": {\"Traveler\": \"Lookup\", \"Date\": \"Date\", \"Destination\": \"Text\", \"Price\": \"Currency\"}}"},
    {"prompt": "A task tracking tool", "schema": "{\"Task\": {\"TaskName\": \"Text\", \"AssignedTo\": \"Lookup\", \"DueDate\": \"Date\", \"Priority\": \"Picklist\"}}"},
    {"prompt": "A donation tracking app", "schema": "{\"Donation\": {\"Donor\": \"Lookup\", \"Amount\": \"Currency\", \"Date\": \"Date\", \"Purpose\": \"Textarea\"}}"},
    {"prompt": "A project management tool", "schema": "{\"Project\": {\"Name\": \"Text\", \"StartDate\": \"Date\", \"EndDate\": \"Date\", \"Budget\": \"Currency\", \"Manager\": \"Lookup\"}}"},
    {"prompt": "A job application tracking system", "schema": "{\"Application\": {\"Name\": \"Text\", \"StartDate\": \"Date\", \"EndDate\": \"Date\", \"Budget\": \"Currency\", \"Manager\": \"Lookup\"}}"},
    {"prompt": "A time tracking tool", "schema": "{\"TimeTracking\": {\"TaskName\": \"Text\", \"StartTime\": \"Time\", \"EndTime\": \"Time\", \"TotalHours\": \"Number\"}}"},
    {"prompt": "A payroll management system", "schema": "{\"Payroll\": {\"EmployeeName\": \"Text\", \"EmployeeID\": \"Text\", \"Month\": \"Month\", \"GrossSalary\": \"Currency\", \"Deductions\": \"Currency\", \"NetSalary\": \"Currency\"}}"},
    {"prompt": "A recruitment management system", "schema": "{\"Recruitment\": {\"CandidateName\": \"Text\", \"Position\": \"Text\", \"ApplicationDate\": \"Date\", \"Status\": \"Picklist\"}}"},
    {"prompt": "A talent management system", "schema": "{\"Talent\": {\"EmployeeName\": \"Text\", \"Skills\": \"Textarea\", \"Experience\": \"Number\", \"Performance\": \"Picklist\"}}"},
    {"prompt": "An HR management system", "schema": "{\"HR\": {\"EmployeeID\": \"Text\", \"Name\": \"Text\", \"Department\": \"Text\", \"Role\": \"Text\", \"JoiningDate\": \"Date\", \"Salary\": \"Currency\"}}"},
    {"prompt": "A music streaming app", "schema": "{\"Song\": {\"Title\": \"Text\", \"Artist\": \"Text\", \"Album\": \"Text\", \"Genre\": \"Picklist\", \"ReleaseDate\": \"Date\"}}"},
    {"prompt": "A video sharing platform", "schema": "{\"Video\": {\"Title\": \"Text\", \"Description\": \"Textarea\", \"UploadDate\": \"Date\", \"Views\": \"Number\", \"Uploader\": \"Lookup\"}}"},
    {"prompt": "An online learning platform", "schema": "{\"Course\": {\"CourseName\": \"Text\", \"Instructor\": \"Lookup\", \"Duration\": \"Number\", \"Price\": \"Currency\", \"Rating\": \"Number\"}}"},
    {"prompt": "A social media platform", "schema": "{\"Post\": {\"Content\": \"Text\", \"Author\": \"Lookup\", \"Timestamp\": \"Date\", \"Likes\": \"Number\", \"Comments\": \"Textarea\"}}"},
    {"prompt": "A restaurant ordering system", "schema": "{\"Order\": {\"OrderID\": \"Number\", \"TableNumber\": \"Text\", \"Items\": \"Picklist\", \"TotalAmount\": \"Currency\", \"Status\": \"Picklist\"}}"},
    {"prompt": "A delivery tracking system", "schema": "{\"Delivery\": {\"TrackingNumber\": \"Text\", \"Customer\": \"Lookup\", \"Status\": \"Picklist\", \"EstimatedDelivery\": \"Date\"}}"}
    # Add the rest of your data here...
]

# Extract prompts and schemas
prompts = [entry["prompt"] for entry in data]
schemas = [entry["schema"] for entry in data]

# Vectorizer (Increased ngram_range, and use sublinear_tf for better term weighting)
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 4), sublinear_tf=True, max_features=10000)
prompt_features = vectorizer.fit_transform(prompts)

# Label Encoder
label_encoder = LabelEncoder()
schema_labels = label_encoder.fit_transform(schemas)

# Train the Logistic Regression model
classifier = LogisticRegression(max_iter=1000, solver='liblinear')  # You can use 'liblinear' for small datasets
classifier.fit(prompt_features, schema_labels)

# Save the trained model, label encoder, and vectorizer
with open(model_path, 'wb') as f:
    pickle.dump(classifier, f)

with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training complete and models saved!")
