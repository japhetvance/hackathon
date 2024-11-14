from datetime import datetime
import json
import openai
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import streamlit as st
import os
import base64
import io
import requests
load_dotenv()

st.set_page_config(layout='wide')

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

if 'show_eligibility_checker' not in st.session_state:
    st.session_state.show_eligibility_checker = False

if "show_eligibility_result" not in st.session_state:
    st.session_state.show_eligibility_result = False

if "show_loan_application_form" not in st.session_state:
    st.session_state.show_loan_application_form = False

if "show_loan_application_results" not in st.session_state:
    st.session_state.show_loan_application_results = False


# Fetch data from APi
@st.cache_data(show_spinner=False)
def fetch_data():
    provinces_api_url = "https://psgc.gitlab.io/api/regions/"
    provinces = requests.get(provinces_api_url)
    provinces.raise_for_status()

    cities_api_url = "https://psgc.gitlab.io/api/regions/130000000/cities/"
    cities = requests.get(cities_api_url)
    cities.raise_for_status()

    return provinces.json(), cities.json()

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

def extract_raw_text_from_img_openai_all(image_bytes, document_type, has_middle_name):
    """Extracts text from the image using GPT-4o (OpenAI Vision) with structured extraction instructions."""
    image = Image.open(io.BytesIO(image_bytes))

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = encode_image(buffered.getvalue())

    # Adjust prompt based on document type
    middle_name_instructions = {
        True: (
                "- Middle Name: If the name format is 'Last, First Middle' and the middle name checkbox is ticked, "
                "treat the last word after the first name as the middle name. Do not skip the middle name unless it is clearly missing. "
                "For example, if the name is 'PAMONAG, JAPHET VANCE MACESAR', extract 'JAPHET VANCE' as First Name, 'MACESAR' as Middle Name, and 'PAMONAG' as Last Name."),
        False: (
                "- Middle Name: If the middle name checkbox is not ticked, classify everything after the last name as part of the first name. "
                "Do not extract a middle name; return **Not Found** for the middle name field.")
    }

    document_type_instructions = {
        "ID": (f"""You are receiving an image of a government-issued primary ID from the Philippines. The ID will contain the following fields, which you should extract with precision:
                    - First Name (labeled as 'Pangalan' or 'Given Names')
                    {middle_name_instructions}
                    - Last Name (labeled as 'Apelyido' or 'Surname')
                    - Birth Date (labeled as 'Petsa ng kapanganakan' or 'Date of birth' use this format 'YYYY-MM-DD')

                    When the format is "Last, First Middle":
                    - If the middle name checkbox is ticked, treat the last word after the first name as the middle name. Do not skip the middle name unless it is clearly missing.
                    - If the middle name checkbox is not ticked, classify everything after the last name as part of the first name, and do not extract a middle name. Return **Not Found** for the middle name field.

                    Example of expected output:
                    First Name: <Extracted First Name>
                    Middle Name: <Extracted Middle Name or 'Not Found'>
                    Last Name: <Extracted Last Name>
                    Birth Date: <Extracted Birth Date or 'Not Found'>"""),

        "DTI": (f"""You are receiving an image of a Department of Trade and Industry (DTI) Certificate from the Philippines. The certificate will contain the following fields, which you should extract with precision:
                    - Business Name (labeled as 'Business Name')
                    - Business Owner (labeled as 'Issued to')
                    - Business Expiry Date (labeled as 'valid from')

                    Example of expected output:
                    Business Name: <Extracted Business Name>
                    Business Owner: <Extracted Business Owner>
                    Business Expiry Date: <Extracted Business Expiry Date>"""),

        "ITR": (f"""You are receiving an image of an Income Tax Return (ITR) from the Philippines. The ITR will contain the following fields, which you should extract with precision:
                    - Taxpayer Name (labeled as 'Taxpayer Name')
                    - Taxpayer TIN (labeled as 'Taxpayer TIN' or 'TIN')
                    - Taxpayer Address (labeled as 'Taxpayer Address' or 'Registered Address')
                    - RDO Code (labeled as 'RDO Code')
                    - Zip Code (labeled as 'Zip Code' containing 3 digits for example '000')
                    - Date of Birth (labeled as 'Date of Birth')
                    - Telephone Number (labeled as 'Telephone Number')
                    - Line of Business (labeled as 'Line of Business')
                    - Method of Deduction (labeled as 'Method of Deduction' options are 'Itemized Deduction' or 'Optional Standard Deduction' marked with an x)
                    - Revenue (labeled as 'Sales/Revenues/Receipts/Fees' or '26A)
                    - Total Revenue (labeled as 'Total' or '28A')
                    - Gross Income (labeled as 'Gross Income from Operation' or '32A')
                    - Total Deductions (labeled as 'Total Deductions' or '33A')
                    - Taxable Income (labeled as 'Taxable Income' or '34A')
                    - Taxable Income to Date (labeled as 'Taxable Income to Date' or '36A')
                    - Total Tax Due (labeled as 'Total Tax Due' or '37A')
                    - Tax Payment (labeled as 'Tax Payment' or '39A')
                    - Total Amount Payable (labeled as 'Total Amount Payable' or '41A')


                    Example of expected output:
                    Taxpayer Name: <Extracted Taxpayer Name>
                    Taxpayer TIN: <Extracted Taxpayer TIN>
                    Taxpayer Address: <Extracted Taxpayer Address>
                    RDO Code: <Extracted RDO Code>
                    Zip Code: <Extracted Zip Code>
                    Date of Birth: <Extracted Date of Birth>
                    Telephone Number: <Extracted Telephone Number>
                    Line of Business: <Extracted Line of Business>
                    Method of Deduction: <Extracted Method of Deduction>
                    Revenue: <Extracted Revenue>
                    Total Revenue: <Extracted Total Revenue>
                    Gross Income: <Extracted Gross Income>
                    Total Deductions: <Extracted Total Deductions>
                    Taxable Income: <Extracted Taxable Income>
                    Taxable Income to Date: <Extracted Taxable Income to Date>
                    Total Tax Due: <Extracted Total Tax Due>
                    Tax Payment: <Extracted Tax Payment>
                    Total Amount Payable: <Extracted Total Amount Payable>"""),

        "Bank_Statement": (f"""You are receiving an image of a Bank Statement from the Philippines. The Bank Statement will contain the following fields, which you should extract with precision:
                    - Bank Name (labeled as 'Bank Name' and usually located at the top of the statement)
                    - Bank Branch (labeled as 'Branch' or 'Branch Name' and usually located below the Bank Name)
                    - Account Holder Name (labeled as 'Account Name' or 'Account Holder Name')
                    - Account Holder Address (labeled as 'Account Address' or 'Account Holder Address')
                    - Account Number (labeled as 'Primary Account Number')
                    - Account Type (labeled as 'Account Type' or 'Type of Account')
                    - Statement Date (labeled as 'Statement Date' or 'Date')
                    - Other Account Type (labeled as 'Deposits & Other Credits' or 'Description')
                    - Other Account Date Credited (labeled as 'Date Credited' or 'Date')


                    Example of expected output:
                    Bank Name: <Extracted Bank Name>
                    Bank Branch: <Extracted Bank Branch>
                    Account Holder Name: <Extracted Account Holder Name>
                    Account Holder Address: <Extracted Account Holder Address>
                    Account Number: <Extracted Account Number>
                    Account Type: <Extracted Account Type>
                    Statement Date: <Extracted Statement Date>
                    Other Account Type: <Extracted Other Account Type>
                    Other Account Date Credited: <Extracted Other Account Date Credited>""")

    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""
                    {document_type_instructions.get(document_type, "Invalid document type. Please provide a valid document type.")}
                    Please output only the extracted text, no need for explanations or other conversational responses.
                    """.strip()},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ],
        max_tokens=2048
    )

    raw_text = response.choices[0].message.content
    return raw_text


def eligibility_checker_form():
    fetch_data()
    provinces, cities = fetch_data()

    if "show_form" not in st.session_state:
        st.session_state.show_form = False

    provinces_names = ["--none selected--"] + sorted([province["name"] for province in provinces])
    cities_names = ["--none selected--"] + sorted([city["name"] for city in cities])

    with open('extracted_data.json', 'r') as json_file:
        data = json.load(json_file)

    if st.session_state.show_form:
        with st.form("eligibility_form"):
            st.write("Please fill in the form below to check your eligibility for a BPI Personal Loan.")
            col1, col2, col3 = st.columns(3)
            with col1:
                first_name = st.text_input("First Name", value=data.get("First Name", ""))
            with col2:
                middle_name = st.text_input("Middle Name", value=data.get("Middle Name", ""))
            with col3:
                last_name = st.text_input("Last Name", value=data.get("Last Name", ""))
            email_address = st.text_input("Email Address", value=data.get("Email", ""))
            province = st.selectbox("Business Address - Province", provinces_names, index=13)
            city = st.selectbox("Business Address - City", cities_names, index=13)
            date_of_birth_str = data.get("Birth Date", "")
            if date_of_birth_str:
                date_of_birth = datetime.strptime(date_of_birth_str, "%Y-%m-%d")
            else:
                date_of_birth = None
            date_of_birth = st.date_input("Date of Birth", value=date_of_birth)
            contact_method = st.radio("Preferred Contact Method", ["Landline Number", "Mobile Number"])
            contact_number = st.text_input("Contact Number", value=data.get("Telephone Number", ""))
            st.write("---")
            existing_depositor = st.radio("Are you an existing depositor of BPI?", ["Yes", "No"])
            how_did_you_find_out = st.text_input("How did you find out about BPI Ka-Negosyo Loans?")
            st.write("---")
            nationality = st.radio("What is your Nationality?", ["Filipino", "Permanent Resident", "Non-Resident"], index=0)
            loan_purpose = st.selectbox("What is the purpose of your loan?",
                                        ["--none selected--", "Working Capital", "Capital to start additional business",
                                         "Business expansion", "Acquisition of Trucks/Vehicles/Equipment",
                                         "Acquisition of Real Estate", "Construction", "Renovation", "Franchise Financing",
                                         "Trade Related Activities", "Loan Takeout"])
            loan_amount = st.text_input("How much would you like to loan?")
            loan_type = st.selectbox("What Ka-Negosyo loan would you like to apply for?",
                                     ["--none selected--", "Ka-Negosyo Credit Line", "Ka-Negosyo Ready Loan",
                                      "Ka-Negosyo SME Loan", "Property Acquisition Loan"])
            st.write("---")
            business_type = st.radio("Select an option that best represents your business",
                                     ["Individual", "Sole Proprietorship", "Professional Engaging in Business",
                                      "One Person Corporation", "Partnership/Corporation"])
            years_in_business = st.text_input("How many years has your company been in business?")
            business_industry = st.text_input("What industry does your business belong to?", value=data.get("Line of Business", ""))
            gross_monthly_income = st.text_input("What is your company's total gross monthly income?", value=data.get("Gross Income", ""))
            loan_timeline = st.selectbox("How soon would your company require the loan?",
                                         ["--none selected--", "Within a month", "In 1-3 months", "3-6 months",
                                          "More than 6 months", "Not yet confirmed"])
            submitted = st.form_submit_button("Submit")
            if submitted:
                if not all([first_name, middle_name, last_name, email_address, province != "--none selected--", city != "--none selected--",
                            date_of_birth, contact_number, how_did_you_find_out, loan_purpose != "--none selected--",
                            loan_amount, loan_type != "--none selected--", years_in_business, business_industry,
                            gross_monthly_income, loan_timeline != "--none selected--"]):
                    st.error("Please fill in all the fields.")
                else:
                    st.success("Form submitted successfully!")
                    st.session_state.form_data = {
                        "first_name": first_name,
                        "middle_name": middle_name,
                        "last_name": last_name,
                        "email_address": email_address,
                        "province": province,
                        "city": city,
                        "date_of_birth": date_of_birth,
                        "contact_method": contact_method,
                        "contact_number": contact_number,
                        "existing_depositor": existing_depositor,
                        "how_did_you_find_out": how_did_you_find_out,
                        "nationality": nationality,
                        "loan_purpose": loan_purpose,
                        "loan_amount": loan_amount,
                        "loan_type": loan_type,
                        "business_type": business_type,
                        "years_in_business": years_in_business,
                        "business_industry": business_industry,
                        "gross_monthly_income": gross_monthly_income,
                        "loan_timeline": loan_timeline
                    }
                    st.session_state.show_form = False
                    st.session_state.show_eligibility_result = True
                    st.rerun()

    elif st.session_state.show_eligibility_result:
        eligibility_results()

    else:
        st.markdown("""Through your expressed consent, you acknowledge and agree that your information may be processed and shared within BPI to communicate with you regarding marketing communications, programs, products and services of the Bank that you may find interesting and relevant. 
                    <br><br>
                    In compliance with the Data Privacy Act (R.A.10173), the personal data collected is treated with confidentiality and will only be retained solely for the fulfillment of the aforementioned purposes. To know more about how we process your personal data, please refer to BPI’s Privacy Policy.""",
                    unsafe_allow_html=True
                    )
        consent = st.checkbox("I consent to the above")
        st.write("---")
        st.write("### Magic Fill-Out: Snap, Upload, and Watch Your Forms Fill Themselves!")
        st.write("Say goodbye to tedious form filling! With Magic Fill-Out, simply upload a photo of your credentials, and let our smart tool do the rest. In seconds, all your information is seamlessly entered into the form fields, saving you time and hassle. It's like magic—just faster!")
        col1, col2 = st.columns(2)
        with col1:
            id_upload = st.file_uploader("Upload Government Valid ID here", type=["jpg", "jpeg", "png"])
            dti_upload = st.file_uploader("Upload DTI Certificate here", type=["jpg", "jpeg", "png"])
        with col2:
            itr_upload = st.file_uploader("Upload ITR here", type=["jpg", "jpeg", "png"])
            bank_statement_upload = st.file_uploader("Upload Bank Statement here", type=["jpg", "jpeg", "png"])

        if id_upload:
            image_bytes = id_upload.getbuffer()
            output = extract_raw_text_from_img_openai_all(image_bytes, "ID", has_middle_name=True)
            lines = output.splitlines()
            info_dict = {}

            for line in lines:
                key, value = line.split(": ")
                info_dict[key] = value

            data.update(info_dict)
            # Write the dictionary to a JSON file
            with open('extracted_data.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)

        if dti_upload:
            image_bytes = dti_upload.getbuffer()
            output = extract_raw_text_from_img_openai_all(image_bytes, "DTI", has_middle_name=False)
            lines = output.splitlines()
            info_dict = {}

            for line in lines:
                key, value = line.split(": ")
                info_dict[key] = value

            data.update(info_dict)
            # Write the dictionary to a JSON file
            with open('extracted_data.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)

        if itr_upload:
            image_bytes = itr_upload.getbuffer()
            output = extract_raw_text_from_img_openai_all(image_bytes, "ITR", has_middle_name=False)
            lines = output.splitlines()
            info_dict = {}

            for line in lines:
                key, value = line.split(": ")
                info_dict[key] = value

            data.update(info_dict)
            # Write the dictionary to a JSON file
            with open('extracted_data.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)

        if bank_statement_upload:
            image_bytes = bank_statement_upload.getbuffer()
            output = extract_raw_text_from_img_openai_all(image_bytes, "Bank_Statement", has_middle_name=False)
            lines = output.splitlines()
            info_dict = {}

            for line in lines:
                key, value = line.split(": ")
                info_dict[key] = value

            data.update(info_dict)
            # Write the dictionary to a JSON file
            with open('extracted_data.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)

        if st.button("Next"):
            if not consent:
                st.error("Please consent to the above before proceeding.")
            else:
                st.session_state.show_form = True
                st.rerun()

def eligibility_results():
    # Main content
    st.markdown("<h2 style='text-align: center; color: black;'>Good news! You are</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 48px; color: #4CAF50;'>Eligible</h1>",
                unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align: center; color: gray;'>
        Thank you for taking the time to share your information with us.<br>
        You may proceed with your application by clicking the button below and keying in the Unique ID sent to your registered email.
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        if st.button("Start Loan Application", use_container_width=True):
            st.session_state.show_eligibility_result = False
            st.session_state.show_eligibility_checker = False
            st.session_state.button_clicked = False
            st.rerun()

    # Additional contact information
    st.markdown(
        "<h4 style='text-align: center; color: black; margin-top: 30px;'>For concerns and inquiries, please contact</h4>",
        unsafe_allow_html=True)

    # Customer support button
    st.markdown("""
    <div style='text-align: center;'>
        <button style='background-color: #FFFFFF; border: 1px solid #DCDCDC; padding: 10px 20px; font-size: 18px; cursor: pointer;'>
            Business Banking Customer Support
        </button>
    </div>
    """, unsafe_allow_html=True)

def loan_application_form():
    with open('extracted_data.json', 'r') as json_file:
        data = json.load(json_file)

    if 'show_application_form' not in st.session_state:
        st.session_state.show_application_form = True
        st.title("Business Loan Application Form")

    if st.session_state.show_application_form:
        with st.form("loan_application_form"):
            col1, col2 = st.columns(2)
            with col1:
                loan_type = st.radio("Loan Type", ["New Application", "Renewal", "Additional Loan", "Restructuring"], index=None)
            with col2:
                previous_application = st.radio("In case of loan renewal or restructuring, are there any update from the previous application?", ["Yes", "No"], index=None)
            business_type = st.radio("Business Type", ["Individual", "Sole Proprietorship"], index=None)
            st.markdown("---")
            st.markdown("### **A. BORROWER AND BUSINESS INFORMATION**")
            st.markdown("---")
            full_name = st.text_input("Name of the Borrower")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                civil_status = st.radio("Civil Status", ["Single", "Married", "Widowed", "Separated", "Annuled"], index=None)
            with col2:
                birth_date = st.date_input("Birth Date", value=None)
            with col3:
                place_of_birth = st.text_input("Place of Birth")
            with col4:
                citizenship = st.text_input("Citizenship")
            with col5:
                sex = st.radio("Sex", ["Male", "Female"] , index=None)

            col1, col2 = st.columns([3, 1])
            with col1:
                spouse_name = st.text_input("Spouse Name")
            with col2:
                spouse_birth_date = st.date_input("Spouse Birth Date", value=None)
            home_address = st.text_input("Home Address")
            col1, col2, col3 = st.columns([2,1,1])
            with col1:
                home_ownership = st.radio("Home Address Ownership", ["Owned (unencumbered)", "Owned (mortgaged)", "Rented", "Living with Relatives"], index=None)
            with col2:
                years_of_stay = st.text_input("Years of Stay")
            with col3:
                telephone_number = st.text_input("Telephone Number")
            col1, col2 = st.columns([1, 2])
            with col1:
                mobile_number = st.text_input("Mobile Number")
            with col2:
                email_address = st.text_input("Email Address")
            col1, col2, col3 = st.columns(3)
            with col1:
                tin = st.text_input("TIN")
            with col2:
                philsys_id = st.text_input("PhilSys ID")
            with col3:
                other_id = st.text_input("Other Government-Issued ID")
            mothers_maiden_name = st.text_input("Mother's Maiden Name")
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                business_name = st.text_input("Registered Business Name")
            with col2:
                years_of_business = st.text_input("Years in Operation")
            with col3:
                num_of_branches = st.text_input("Number of Branches")
            col1, col2 = st.columns([3, 1])
            with col1:
                similar_to_home_address = st.radio("Is the Business Address similar to Home Address?", ["Yes", "No"], index=None)
                principal_business_address = st.text_input("Principal Business Address")
            with col2:
                business_address_ownership = st.radio("Business Address Ownership", ["Owned (unencumbered)", "Owned (mortgaged)", "Rented"], index=None)
            col1, col2 = st.columns(2)
            with col1:
                website = st.text_input("Website/social media page")
            with col2:
                indicate_business_has = st.radio("Indicate weather the business has", ["Female Manager", "Female head officer of operations"] , index=None)
            col1, col2 = st.columns([1,2])
            with col1:
                nature_of_business = st.text_input("Nature of Business")
            with col2:
                business_activity = st.text_input("Please specify the business activity")
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("Business Registration")
            with col2:
                st.markdown("Date of Business Registration")
            with col3:
                st.markdown("Expiry Date of Registration")
            with col4:
                st.markdown("Registration Number")
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                dti = st.checkbox("DTI")
            with col2:
                dti_date = st.date_input("DTI Date", value=None)
            with col3:
                dti_expiry = st.date_input("DTI Expiry", value=None)
            with col4:
                dti_reg_num = st.text_input("DTI Reg. Number")
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                bir = st.checkbox("BIR")
            with col2:
                bir_date = st.date_input("BIR Date", value=None)
            with col3:
                bir_expiry = st.date_input("BIR Expiry", value=None)
            with col4:
                bir_reg_num = st.text_input("BIR Reg. Number")
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                firm_size = st.radio("Firm Size", ["Micro (not more than Php 3M)", "Small (Php 3m to 15M)", "Medium (Php 15M to 100M)"], index=None)
            with col2:
                annual_sales_revenue = st.text_input("Annual Sales Revenue")
                number_of_employees = st.text_input("Number of Employees")

            st.markdown("---")
            st.markdown("### **B. LOAN APPLICATION INFORMATION**")
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                loan_amount = st.text_input("Loan Amount")
            with col2:
                repayment_frequency = st.radio("Repayment Frequency", ["Weekly", "Monthly", "Quarterly", "Annually", "Lump sum"], index=None)

            col1, col2, col3 = st.columns(3)
            with col1:
                loan_facility = st.radio("Loan Facility", ["Credit Line", "Term Loan", "Others"], index=None)
            with col2:
                loan_purpose = st.radio("Loan Purpose", ["Working Capital", "Capital to start additional business", "Business expansion", "Acquisition of Trucks/Vehicles/Equipment", "Acquisition of Real Estate", "Construction", "Renovation", "Franchise Financing", "Trade Related Activities", "Loan Takeout"], index=None)
            with col3:
                tenor_months = st.text_input("Tenor (Months)")
            loan_type = st.radio("Loan Type", ["Unsecured Loan", "Secured Loan"], index=None)
            collateral = st.text_input("If secured, collateral offered:")

            st.markdown("---")
            st.markdown("### **C. FINANCIAL INFORMATION**")
            st.markdown("---")
            source_of_repayment = st.radio("Source of Repayment", ["Revenue", "Asset Sale", "Savings and/or Investment", "Inheritance", "Salary/Allowance", "Others"], index=None)
            st.markdown("---")
            st.markdown("**Existing Deposit and E-money Account**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("Name of Financial Institution")
            with col2:
                st.markdown("Type of Account")
            with col3:
                st.markdown("Year Opened")
            with col4:
                st.markdown("Type of Account Ownership")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                deposit_name = st.text_input("Deposit Name")
            with col2:
                deposit_type = st.radio("Deposit Type", ["Savings", "Checking", "E-wallet", "Others"], index=None)
            with col3:
                deposit_year_opened = st.text_input("Year Opened")
            with col4:
                deposit_ownership = st.radio("Deposit Ownership", ["Personal", "Business/Merchant"], index=None)
            st.markdown("---")
            st.markdown("**Existing Loans**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("Name of Financial Institution")
            with col2:
                st.markdown("Loan Amount")
            with col3:
                st.markdown("Date Granted")
            col1, col2, col3 = st.columns(3)
            with col1:
                loan_name = st.text_input("Loan Name")
            with col2:
                existing_loan_amount = st.text_input("Existing Loan Amount")
            with col3:
                loan_date_granted = st.date_input("Date Granted", value=None)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("Maturity Date")
            with col2:
                st.markdown("Outstanding Balance")
            with col3:
                st.markdown("Collateral/s Offered")
            col1, col2, col3 = st.columns(3)
            with col1:
                loan_maturity_date = st.date_input("Maturity Date", value=None)
            with col2:
                outstanding_balance = st.text_input("Outstanding Balance")
            with col3:
                collateral_offered = st.text_input("Collateral Offered")
            st.markdown("---")
            st.markdown("**Existing Credit Cards**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("Name of Financial Institution")
            with col2:
                st.markdown("Credit Limit")
            with col3:
                st.markdown("Outstanding Balance")
            with col4:
                st.markdown("Type of Ownership")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                credit_name = st.text_input("Credit Name")
            with col2:
                credit_limit = st.text_input("Credit Limit")
            with col3:
                credit_outstanding_balance = st.text_input("Card Outstanding Balance")
            with col4:
                credit_ownership = st.radio("Credit Ownership", ["Personal", "Business"], index=None)

            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state.show_application_form = False
                st.session_state.show_loan_application_results = True
                st.rerun()

def loan_application_results(loan_type):
    # Main content
    st.markdown("<h2 style='text-align: center; color: black;'>Congratulations!</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 48px; color: #4CAF50;'>Your Loan Application is Successful</h1>",
                unsafe_allow_html=True)

    st.markdown(f"""
        <p style='text-align: center; color: gray;'>
            Thank you for applying to our <strong>{loan_type}</strong>. Your application has been successfully submitted.
            <br>
            Our team will review it, and you will receive an email with further instructions within the next few business days.
        </p>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        if st.button("Return Home", use_container_width=True):
            st.session_state.show_loan_application_form = False
            st.session_state.show_loan_application_results = False
            st.session_state.button_clicked = False
            st.rerun()

    # Additional contact information
    st.markdown(
        "<h4 style='text-align: center; color: black; margin-top: 30px;'>For concerns and inquiries, please contact</h4>",
        unsafe_allow_html=True)

    # Customer support button
    st.markdown("""
        <div style='text-align: center;'>
            <button style='background-color: #FFFFFF; border: 1px solid #DCDCDC; padding: 10px 20px; font-size: 18px; cursor: pointer;'>
                Business Banking Customer Support
            </button>
        </div>
        """, unsafe_allow_html=True)

def homepage():

    if "home" not in st.session_state:
        st.session_state.home = True

    if st.session_state.home:
        st.markdown("<h1 style='text-align: center; color: black;'>Welcome to BPI Ka-Negosyo Loans</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p style='text-align: center; color: gray;'>
            We are here to help you grow your business. Whether you are looking to expand your operations, 
            purchase new equipment, or simply need working capital, we have the right loan for you.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p style='text-align: center; color: gray;'>
            To get started, please select an option below:
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("<h2 style='text-align: center; color: black;'>What would you like to do today?</h2>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col2:
            if st.button("Eligibility Checker", key="eligibility_checker", use_container_width=True):
                st.session_state.show_eligibility_checker = True
                st.session_state.show_loan_application_form = False
                st.session_state.show_loan_application_results = False
                st.session_state.home = False
                st.rerun()

        with col3:
            if st.button("Loan Application Form", key="loan_application_form" , use_container_width=True):
                st.session_state.show_eligibility_checker = False
                st.session_state.show_loan_application_form = True
                st.session_state.show_loan_application_results = False
                st.session_state.home = False
                st.rerun()

def main():
    homepage()

    if st.session_state.show_eligibility_checker:
        eligibility_checker_form()

    if st.session_state.show_loan_application_form:
        loan_application_form()

    if st.session_state.show_eligibility_result:
        eligibility_results()

    if st.session_state.show_loan_application_results:
        loan_application_results("Ka-Negosyo Credit Line")

if __name__ == "__main__":
    main()