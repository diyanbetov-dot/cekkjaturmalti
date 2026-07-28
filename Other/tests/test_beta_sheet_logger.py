import os
import unittest
from unittest.mock import patch, MagicMock
from Essentials.app import app
from Essentials.helpers import beta_sheet_logger

class TestBetaSheetLogger(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    @patch.dict(os.environ, {"SPELLCHECK_BETA_LOGGING": "false"}, clear=True)
    def test_logging_disabled_by_default(self):
        self.assertFalse(beta_sheet_logger.is_logging_enabled())
        self.assertFalse(beta_sheet_logger.create_log("id1", "input", "output"))
        self.assertFalse(beta_sheet_logger.update_choice("id1", "evt1", "token", ["opt1"], "opt1", "final"))

    @patch.dict(os.environ, {
        "SPELLCHECK_BETA_LOGGING": "true",
        "SPELLCHECK_LOG_URL": "https://script.google.com/macros/s/test/exec",
        "SPELLCHECK_LOG_SECRET": "test_secret_123"
    })
    @patch("urllib.request.urlopen")
    def test_create_log_payload_and_execution(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = beta_sheet_logger.create_log(
            log_id="550e8400-e29b-41d4-a716-446655440000",
            input_text="Alaqli l bieb",
            initial_output="Għalaqli l-bieb."
        )

        self.assertTrue(res)
        self.assertTrue(mock_urlopen.called)
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "https://script.google.com/macros/s/test/exec")

    @patch.dict(os.environ, {
        "SPELLCHECK_BETA_LOGGING": "true",
        "SPELLCHECK_LOG_URL": "https://script.google.com/macros/s/test/exec",
        "SPELLCHECK_LOG_SECRET": "test_secret_123"
    })
    @patch("urllib.request.urlopen")
    def test_update_choice_payload(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = beta_sheet_logger.update_choice(
            log_id="550e8400-e29b-41d4-a716-446655440000",
            event_id="evt_001",
            token="Għalaqli",
            suggestions=["Għalaqli", "Agħlaqli"],
            chosen="Agħlaqli",
            final_output="Agħlaqli l-bieb."
        )

        self.assertTrue(res)
        self.assertTrue(mock_urlopen.called)

    @patch.dict(os.environ, {
        "SPELLCHECK_BETA_LOGGING": "true",
        "SPELLCHECK_LOG_URL": "https://script.google.com/macros/s/test/exec",
        "SPELLCHECK_LOG_SECRET": "super_secret_key"
    })
    @patch("urllib.request.urlopen")
    def test_check_text_returns_log_id_and_fails_open(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Google Apps Script network timeout")

        response = self.client.post("/check-text", json={"text": "Moħħok hemm."})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("log_id", data)
        self.assertTrue(len(data["log_id"]) > 0)
        self.assertIn("corrected_text", data)
        # Verify secret is never exposed in response JSON
        self.assertNotIn("super_secret_key", str(data))

    def test_log_suggestion_choice_missing_log_id_rejected(self):
        response = self.client.post("/log-suggestion-choice", json={
            "token": "test",
            "chosen": "choice"
        })
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_log_suggestion_choice_invalid_suggestions_rejected(self):
        response = self.client.post("/log-suggestion-choice", json={
            "log_id": "test_log_1",
            "token": "test",
            "suggestions": "not a list",
            "chosen": "choice"
        })
        self.assertEqual(response.status_code, 400)

    @patch.dict(os.environ, {
        "SPELLCHECK_BETA_LOGGING": "true",
        "SPELLCHECK_LOG_URL": "https://script.google.com/macros/s/test/exec",
        "SPELLCHECK_LOG_SECRET": "super_secret_key"
    })
    @patch("urllib.request.urlopen")
    def test_log_suggestion_choice_successful(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response

        response = self.client.post("/log-suggestion-choice", json={
            "log_id": "550e8400-e29b-41d4-a716-446655440000",
            "event_id": "evt_123",
            "token": "Għalaqli",
            "suggestions": ["Għalaqli", "Agħlaqli"],
            "chosen": "Agħlaqli",
            "final_output": "Agħlaqli l-bieb f'wiċċi."
        })

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("log_id"), "550e8400-e29b-41d4-a716-446655440000")
        self.assertNotIn("super_secret_key", str(data))

    @patch.dict(os.environ, {
        "SPELLCHECK_BETA_LOGGING": "true",
        "SPELLCHECK_LOG_URL": "https://script.google.com/macros/s/test/exec",
        "SPELLCHECK_LOG_SECRET": "feedback_secret"
    })
    @patch.object(beta_sheet_logger, "_post_payload", return_value=True)
    def test_submit_feedback_payload(self, mock_post):
        submitted = beta_sheet_logger.submit_feedback(
            email="reader@example.com",
            subject="Kelma mhux rikonoxxuta",
            message="Il-kelma kienet test.",
            screenshot_data_url="data:image/jpeg;base64,/9j/2Q==",
            reported_word="test",
            log_id="log-123",
        )

        self.assertTrue(submitted)
        payload = mock_post.call_args.args[0]
        self.assertEqual(payload["action"], "feedback_report")
        self.assertEqual(payload["email"], "reader@example.com")
        self.assertEqual(payload["reported_word"], "test")
        self.assertEqual(payload["log_id"], "log-123")
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 12.0)

    @patch.dict(os.environ, {"SPELLCHECK_BETA_LOGGING": "false"}, clear=True)
    def test_feedback_route_requires_configured_service(self):
        response = self.client.post("/submit-feedback", json={
            "email": "reader@example.com",
            "subject": "Test",
            "message": "Test message"
        })
        self.assertEqual(response.status_code, 503)

    def test_feedback_route_rejects_invalid_email(self):
        response = self.client.post("/submit-feedback", json={
            "email": "not-an-email",
            "subject": "Test",
            "message": "Test message"
        })
        self.assertEqual(response.status_code, 400)

    @patch.dict(os.environ, {
        "SPELLCHECK_BETA_LOGGING": "true",
        "SPELLCHECK_LOG_URL": "https://script.google.com/macros/s/test/exec",
        "SPELLCHECK_LOG_SECRET": "feedback_secret"
    })
    @patch.object(beta_sheet_logger, "submit_feedback", return_value=True)
    def test_feedback_route_accepts_screenshot(self, mock_submit):
        response = self.client.post("/submit-feedback", json={
            "email": "reader@example.com",
            "subject": "Unrecognized word",
            "message": "Please inspect this word.",
            "screenshot": "data:image/jpeg;base64,/9j/2Q==",
            "reported_word": "kelma",
            "language": "en",
            "log_id": "log-123"
        })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])
        self.assertEqual(mock_submit.call_args.kwargs["reported_word"], "kelma")
        self.assertTrue(
            mock_submit.call_args.kwargs["screenshot_filename"].endswith(".jpg")
        )

if __name__ == "__main__":
    unittest.main()
