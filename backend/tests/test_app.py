import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import BASE_PATH, app, parse_allowed_origins, startup_errors


class DeploymentConfigurationTests(unittest.TestCase):
    def test_render_frontend_is_allowed_to_call_api(self):
        response = TestClient(app).options(
            "/geo_data",
            headers={
                "Origin": "https://arms-trade-dashboard.onrender.com",
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "https://arms-trade-dashboard.onrender.com",
        )

    def test_unconfigured_origin_is_not_allowed_to_call_api(self):
        response = TestClient(app).options(
            "/geo_data",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
            },
        )

        self.assertIsNone(response.headers.get("access-control-allow-origin"))

    def test_cors_origin_override_is_normalized(self):
        self.assertEqual(
            parse_allowed_origins(
                " https://preview.example/ , https://dashboard.example "
            ),
            ["https://preview.example", "https://dashboard.example"],
        )

    def test_health_endpoint_is_available_to_render(self):
        response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_health_endpoint_rejects_a_data_broken_deployment(self):
        with patch.dict(startup_errors, {"trade": "missing"}, clear=True):
            response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"detail": {"status": "degraded", "datasets": ["trade"]}},
        )

    def test_data_path_is_resolved_from_the_application(self):
        self.assertTrue(BASE_PATH.is_dir())
        self.assertTrue((BASE_PATH / "sipri_milex_data_nested.json").is_file())


if __name__ == "__main__":
    unittest.main()
