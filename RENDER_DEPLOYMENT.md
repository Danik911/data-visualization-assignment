# Deploying the Housing Data Dashboard on Render

This guide explains how to deploy the Housing Data Dashboard application on Render.

## Prerequisites

1. A GitHub account
2. A Render account (sign up at [render.com](https://render.com))
3. Google Maps API key

## Deployment Steps

### 1. Push your code to GitHub

If your code is not already on GitHub, you need to create a repository and push your code:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Create a New Web Service on Render

1. Log in to your Render account and navigate to the dashboard
2. Click on "New" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service with these settings:
   - **Name**: housing-data-dashboard (or your preferred name)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:server`

### 3. Set Environment Variables

Add the following environment variable:
- **Key**: `GOOGLE_MAPS_API_KEY`
- **Value**: Your Google Maps API key

### 4. Deploy the Service

Click "Create Web Service" to deploy your application. The initial deployment may take a few minutes.

### 5. Verify the Deployment

Once deployment is complete, Render will provide a URL for your application. Visit this URL to verify that your dashboard is working correctly.

## Troubleshooting

If you encounter issues during deployment:

1. Check the logs in the Render dashboard for error messages
2. Ensure your app.py file correctly defines the server variable
3. Verify that all required dependencies are in requirements.txt
4. Make sure your GOOGLE_MAPS_API_KEY is set correctly in the environment variables

## Maintenance

- You can set up automatic deployments by pushing to your GitHub repository
- Monitor application performance in the Render dashboard 