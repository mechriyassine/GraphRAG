from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from pathlib import Path
import os

def create_sample_pdfs():
    """Create sample PDF files for GraphRAG testing"""
    
    # Create pdfs folder if it doesn't exist
    pdfs_folder = Path("pdfs")
    pdfs_folder.mkdir(exist_ok=True)
    
    # Sample content for different PDFs
    documents = {
        "company_structure.pdf": """
        TechCorp Organizational Structure Report
        
        Executive Leadership:
        Sarah Johnson serves as the Chief Executive Officer (CEO) of TechCorp and has been leading the company for 5 years. She reports directly to the Board of Directors and manages the entire organization.
        
        Department Heads:
        Under Sarah's leadership, there are three key department heads:
        - Michael Chen is the Chief Technology Officer (CTO) and manages all technical operations
        - Lisa Rodriguez serves as the Chief Marketing Officer (CMO) and oversees marketing and sales initiatives  
        - David Kim works as the Chief Financial Officer (CFO) and handles all financial matters and budgeting
        
        Engineering Team:
        The engineering department is led by Michael Chen. Under his supervision are several senior engineers:
        - Anna Petrov leads the AI development team and specializes in machine learning algorithms
        - James Wilson manages the web development team and oversees frontend and backend development
        - Maria Garcia supervises the mobile app development team and coordinates iOS and Android projects
        
        Collaborative Relationships:
        Anna Petrov works closely with Dr. Robert Taylor, who leads the data science team. Dr. Taylor reports to Michael Chen but collaborates extensively with Anna on AI initiatives. The two teams work together on predictive analytics and natural language processing projects.
        
        Marketing Department:
        Lisa Rodriguez manages two key team leads in the marketing department:
        - Jennifer Chang handles all digital marketing campaigns and social media strategy
        - Thomas Anderson manages customer relations and provides customer support services
        
        Cross-Department Collaboration:
        Jennifer Chang frequently collaborates with the engineering teams to understand product features for marketing materials. She works particularly closely with James Wilson on web-related marketing campaigns and user experience improvements.
        
        Financial Operations:
        David Kim supervises the entire accounting team and maintains close working relationships with all department heads for budget planning and financial reporting. He conducts weekly financial review meetings with Sarah Johnson to discuss company performance and strategic investments.
        """,
        
        "project_alpha.pdf": """
        Project Alpha Development Report
        
        Project Overview:
        Project Alpha is TechCorp's flagship AI-powered customer service platform currently in development. This revolutionary system aims to transform how businesses interact with their customers through intelligent automation.
        
        Project Leadership:
        The project is overseen by CEO Sarah Johnson, who provides strategic direction and executive oversight. The technical leadership is managed by CTO Michael Chen, who coordinates all development activities.
        
        Core Development Team:
        Anna Petrov leads the AI development for Project Alpha. Her team is responsible for creating the natural language processing algorithms and machine learning models that power the platform's intelligent responses.
        
        Dr. Robert Taylor supports the project with his data science expertise. He works under Anna's direction for this project, focusing on training data analysis and model optimization. Taylor's team processes customer interaction data to improve the AI's response accuracy.
        
        Technical Infrastructure:
        James Wilson's web development team handles the platform's user interface and API integration. James reports to Michael Chen on technical progress and works closely with Anna Petrov to ensure seamless integration between the AI engine and the web platform.
        
        Mobile Integration:
        Maria Garcia supervises the mobile app integration for Project Alpha. Her team ensures the AI platform works seamlessly across iOS and Android devices. Maria coordinates with both Anna and James to maintain consistency across all platforms.
        
        Marketing Strategy:
        Lisa Rodriguez leads the marketing strategy for Project Alpha's upcoming launch. She works directly with Jennifer Chang to develop marketing campaigns that highlight the platform's AI capabilities.
        
        Customer Success:
        Thomas Anderson provides input on customer needs and pain points based on his experience in customer relations. He reports his findings to both Lisa Rodriguez and the development teams to ensure the product meets market demands.
        
        Budget and Resources:
        David Kim manages the project budget and resource allocation. He works closely with all team leads to ensure proper funding and reports weekly to Sarah Johnson on project expenses and ROI projections.
        """,
        
        "hr_policies.pdf": """
        Human Resources Department Policies and Procedures
        
        HR Department Structure:
        The Human Resources department at TechCorp is led by Director Rachel Foster, who reports directly to CEO Sarah Johnson. Rachel has been with the company for 8 years and oversees all HR operations.
        
        HR Team Members:
        Under Rachel Foster's supervision are three key HR specialists:
        - Kevin Park manages employee recruitment and onboarding processes
        - Diana Wu handles employee relations and conflict resolution
        - Mark Stevens oversees training and professional development programs
        
        Recruitment Process:
        Kevin Park leads all recruitment efforts and works closely with department heads to understand staffing needs. He collaborates regularly with Michael Chen for technical positions, Lisa Rodriguez for marketing roles, and David Kim for finance positions.
        
        Employee Relations:
        Diana Wu serves as the primary point of contact for employee concerns and workplace issues. She works closely with all department managers to address personnel matters and maintain positive workplace culture.
        
        Training and Development:
        Mark Stevens coordinates with team leads across the organization to identify training needs. He works particularly closely with Anna Petrov and Dr. Robert Taylor to develop technical training programs for AI and data science skills.
        
        Performance Reviews:
        Rachel Foster oversees the annual performance review process. She collaborates with all department heads - Sarah Johnson, Michael Chen, Lisa Rodriguez, and David Kim - to ensure consistent evaluation standards across the organization.
        
        Interdepartmental Coordination:
        The HR team maintains regular communication with all departments:
        - Weekly meetings with department heads on staffing needs
        - Monthly check-ins with team leads on employee satisfaction
        - Quarterly reviews with executive leadership on HR metrics
        
        Professional Development:
        Mark Stevens works with external training providers and internal mentors to create development opportunities. Senior employees like Anna Petrov and Dr. Robert Taylor often mentor junior staff members in their respective areas of expertise.
        """,
        
        "quarterly_review.pdf": """
        Q3 2024 Quarterly Business Review
        
        Executive Summary:
        This quarterly review covers TechCorp's performance for Q3 2024, prepared under the direction of CEO Sarah Johnson and reviewed by the executive leadership team.
        
        Financial Performance:
        CFO David Kim reports strong financial results for Q3 2024. Revenue increased by 23% compared to Q2, driven primarily by new client acquisitions and successful product launches. David presented these results to the Board of Directors and received approval for increased R&D investment.
        
        Technology Achievements:
        CTO Michael Chen highlights significant technical milestones achieved this quarter:
        - Project Alpha reached beta testing phase under Anna Petrov's leadership
        - Web platform performance improved by 40% thanks to James Wilson's optimization efforts
        - Mobile app downloads increased by 60% due to Maria Garcia's team improvements
        
        AI and Data Science Progress:
        Anna Petrov's AI development team successfully deployed three new machine learning models in production. Her collaboration with Dr. Robert Taylor's data science team resulted in 25% improvement in prediction accuracy for customer behavior models.
        
        Marketing Success:
        CMO Lisa Rodriguez reports exceptional marketing performance this quarter:
        - Jennifer Chang's digital marketing campaigns generated 45% more leads than previous quarter
        - Thomas Anderson's customer satisfaction scores improved to 94%, up from 87% in Q2
        - Brand awareness increased by 30% across target demographics
        
        Human Resources Updates:
        HR Director Rachel Foster provides staffing and culture updates:
        - Kevin Park successfully recruited 12 new employees across all departments
        - Diana Wu resolved workplace issues with 98% employee satisfaction rate
        - Mark Stevens launched new professional development program with participation from all senior staff
        
        Cross-Department Initiatives:
        Several successful cross-department collaborations occurred this quarter:
        - Anna Petrov and Jennifer Chang worked together on AI-powered marketing automation
        - James Wilson and Thomas Anderson improved customer portal user experience
        - Dr. Robert Taylor and David Kim developed new financial forecasting models
        
        Looking Ahead to Q4:
        CEO Sarah Johnson outlines key priorities for Q4 2024, including Project Alpha's full launch, continued international expansion, and enhanced cross-team collaboration initiatives.
        """
    }
    
    print("📄 Creating sample PDF files for GraphRAG testing...")
    
    for filename, content in documents.items():
        pdf_path = pdfs_folder / filename
        
        try:
            # Create PDF using reportlab
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Split content into paragraphs and add to PDF
            paragraphs = content.strip().split('\n\n')
            for para in paragraphs:
                if para.strip():
                    if para.strip().endswith(':') or len(para.strip().split()) <= 6:
                        # Treat as heading
                        p = Paragraph(para.strip(), styles['Heading2'])
                    else:
                        # Treat as body text
                        p = Paragraph(para.strip(), styles['Normal'])
                    story.append(p)
                    story.append(Spacer(1, 12))
            
            doc.build(story)
            print(f"✅ Created: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to create {filename}: {e}")
    
    print(f"\n🎉 Sample PDFs created in the 'pdfs' folder!")
    print(f"📁 Location: {pdfs_folder.absolute()}")
    print(f"\n🚀 Now you can run your GraphRAG script to process these PDFs!")
    
    # List created files
    pdf_files = list(pdfs_folder.glob("*.pdf"))
    print(f"\n📚 Created {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        size_kb = pdf_file.stat().st_size / 1024
        print(f"  • {pdf_file.name} ({size_kb:.1f} KB)")

def create_simple_text_pdfs():
    """Alternative: Create simple PDFs using basic canvas (no extra dependencies)"""
    
    pdfs_folder = Path("pdfs")
    pdfs_folder.mkdir(exist_ok=True)
    
    documents = {
        "team_structure.pdf": [
            "Team Structure Document",
            "",
            "Alice Johnson is the CEO of our company.",
            "She manages Bob Smith, who is the CTO.", 
            "Bob supervises Charlie Brown, the lead engineer.",
            "Charlie works closely with Diana Prince on AI projects.",
            "Diana reports to Bob but collaborates with Alice on strategy.",
            "",
            "Marketing Team:",
            "Eve Davis leads the marketing department.",
            "She manages Frank Miller, who handles digital campaigns.",
            "Frank works with Charlie on technical marketing materials.",
        ],
        
        "project_status.pdf": [
            "Project Status Report",
            "",
            "Current active projects at our company:",
            "",
            "Project Zeus:",
            "Led by Charlie Brown with support from Diana Prince.",
            "Bob Smith provides technical oversight.",
            "Alice Johnson reviews progress monthly.",
            "",
            "Project Apollo:", 
            "Managed by Frank Miller with Eve Davis supervision.",
            "Charlie Brown assists with technical requirements.",
            "",
            "Project Hermes:",
            "Joint effort between Alice Johnson and Bob Smith.",
            "Diana Prince leads the AI components.",
        ]
    }
    
    print("📄 Creating simple PDF files...")
    
    for filename, lines in documents.items():
        pdf_path = pdfs_folder / filename
        
        try:
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            width, height = letter
            
            y_position = height - 50  # Start near top
            
            for line in lines:
                if y_position < 50:  # Start new page if needed
                    c.showPage()
                    y_position = height - 50
                
                c.drawString(50, y_position, line)
                y_position -= 20
            
            c.save()
            print(f"✅ Created: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to create {filename}: {e}")
    
    print(f"\n🎉 Simple PDFs created!")

if __name__ == "__main__":
    print("🔧 PDF Creator for GraphRAG Testing")
    print("=" * 40)
    
    try:
        # Try to import reportlab for rich PDFs
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        
        create_sample_pdfs()
        
    except ImportError:
        print("📦 reportlab not installed. Creating simple PDFs instead...")
        print("💡 To create richer PDFs, install reportlab: pip install reportlab")
        
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            create_simple_text_pdfs()
        except ImportError:
            print("❌ Cannot create PDFs - reportlab not available")
            print("🔧 Install reportlab first: pip install reportlab")