# Google Sheets Beta Logging Setup Guide

This guide explains how to set up, configure, deploy, and verify the Google Sheets beta logging system for the Maltese Spellchecker.

---

## 1. Expected Google Sheet Structure

The target sheet **must** have a tab named **`Logs`**.

### Column Headings (Order A -> G)

| Column | Heading | Description |
| :--- | :--- | :--- |
| **A** | `Log ID` | Unique UUID generated per spellcheck submission |
| **B** | `Timestamp` | Submission time (format: `DD/Month/YYYY HH:MM:SS`) |
| **C** | `Input` | Exact original text submitted by user |
| **D** | `Initial Output` | Automatic spellchecker result before manual user selection |
| **E** | `Notes` | Manual user suggestion choices appended on separate lines |
| **F** | `Final Output` | Latest text state after any manual suggestion choices |
| **G** | `Processed Event IDs` | Internal tracking list of processed `event_id`s (prevents duplicate logs) |

> [!NOTE]
> The Apps Script will automatically create this header row if the `Logs` tab is empty.

---

## 2. Step-by-Step Setup Instructions

### Step 1: Create a Google Sheet
1. Open [Google Sheets](https://sheets.google.com) and create a new blank spreadsheet.
2. Rename the spreadsheet (e.g., `Maltese Spellchecker Beta Logs`).
3. Rename the first tab at the bottom to **`Logs`**.

### Step 2: Open Google Apps Script Editor
1. In your Google Sheet, click **Extensions** > **Apps Script** in the top menu.
2. Delete any default code in `Code.gs`.
3. Copy the complete code from [`docs/google_sheets_logger.gs`](google_sheets_logger.gs) and paste it into the editor.

### Step 3: Add Script Property (`LOGGING_SECRET`)
1. In the Apps Script left sidebar, click **Project Settings** (the gear icon ⚙️).
2. Scroll down to **Script Properties** and click **Add script property**.
3. Set **Property**: `LOGGING_SECRET`
4. Set **Value**: Generate a strong secret string (e.g., `sk_beta_7f8a9b0c1d2e3f4a5b6c7d8e9f0a`).
5. Click **Save script properties**.

### Step 4: Deploy Web App
1. In the top right corner of Apps Script, click **Deploy** > **New deployment**.
2. Click the gear icon next to **Select type** and choose **Web app**.
3. Set fields:
   - **Description**: `Maltese Spellchecker Logger Web App`
   - **Execute as**: `Me` (your Google account)
   - **Who has access**: `Anyone` (required so Cloud Run can POST to the webhook endpoint)
4. Click **Deploy**.
5. Authorize access when prompted.
6. Copy the generated **Web App URL** (ending in `/exec`). Example:
   `https://script.google.com/macros/s/AKfycbx.../exec`

### Step 5: Configure Environment Variables in Cloud Run
Add the following three environment variables to your Cloud Run deployment or `.env` environment:

```bash
SPELLCHECK_BETA_LOGGING=true
SPELLCHECK_LOG_URL=https://script.google.com/macros/s/YOUR_DEPLOYMENT_ID/exec
SPELLCHECK_LOG_SECRET=your_generated_secret_string
```

### Step 6: Redeploy the Flask Service
Redeploy your Cloud Run container or restart your server so the environment variables take effect.

---

## 3. How to Verify Integration

1. Open your spellchecker web interface.
2. Type a test sentence with an error (e.g., `Alaqli l bieb fwicci`).
3. Click **Iċċekkja l-ortografija** (Check spelling).
4. Check your Google Sheet `Logs` tab:
   - A new row will appear containing the `Log ID`, `Timestamp`, original `Input`, `Initial Output`, and `Final Output`.
5. In the UI, click on a highlighted word suggestion (e.g., change `Għalaqli` to `Agħlaqli`).
6. Check your Google Sheet again:
   - The `Notes` column will update to: `Għalaqli - suggestions: Għalaqli, Agħlaqli - chosen by user: Agħlaqli.`
   - The `Final Output` column will update to: `Agħlaqli l-bieb f'wiċċi.`

---

## 4. Updating Apps Script Code in the Future

If you edit the script in `google_sheets_logger.gs`:
1. Open **Extensions** > **Apps Script**.
2. Update the code.
3. Click **Deploy** > **Manage deployments**.
4. Click the edit pencil icon ✏️ next to the active deployment.
5. Under **Version**, select **New version**.
6. Click **Deploy**.
