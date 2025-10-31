import json
import httpx
import asyncio
from mcp.server.fastmcp import FastMCP

# -------------------------------------------------------------
# MCP Server: project-1
# Tool: api_invoke
# -------------------------------------------------------------
# INPUT:
# {
#   "method": "GET" | "POST" | "PUT" | "DELETE" | "PATCH",
#   "url": "https://api.example.com/endpoint",
#   "headers": { ... },     # optional
#   "params": { ... },      # optional
#   "body": { ... }         # optional
# }
#
# OUTPUT:
# {
#   "status_code": int | null,
#   "headers": { ... },
#   "body": dict | str | null,
#   "error": { "message": str, "type": str } | null
# }
# -------------------------------------------------------------

# Create MCP server
mcp = FastMCP("project-1")

# Configure async HTTP client with 10s timeout
http_client = httpx.AsyncClient(timeout=10.0)

@mcp.tool()
async def api_invoke(
    method: str,
    url: str,
    headers: dict = None,
    params: dict = None,
    body: dict = None
) -> dict:
    """
    Invoke an external API via HTTP request.
    Supports GET, POST, PUT, DELETE, PATCH.
    """

    if not url:
        return {
            "status_code": None,
            "headers": {},
            "body": None,
            "error": {"message": "Missing 'url' in input", "type": "InputError"}
        }

    try:
        response = await http_client.request(
            method=method.upper(),
            url=url,
            headers=headers or {},
            params=params or {},
            json=body or None
        )

        # ✅ Debug log for tracing API activity
        print(f"[APIInvoke] {method.upper()} {url} -> {response.status_code} | params={params} body={body}")

        # Try to parse JSON response body
        try:
            parsed_body = response.json()
        except Exception:
            parsed_body = response.text

        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": parsed_body,
            "error": None
        }


    except httpx.TimeoutException:
        return {
            "status_code": None,
            "headers": {},
            "body": None,
            "error": {"message": "Request timed out", "type": "Timeout"}
        }

    except httpx.RequestError as e:
        return {
            "status_code": None,
            "headers": {},
            "body": None,
            "error": {"message": str(e), "type": "RequestError"}
        }

    except Exception as e:
        return {
            "status_code": None,
            "headers": {},
            "body": None,
            "error": {"message": str(e), "type": "Other"}
        }


if __name__ == "__main__":
    # Run MCP server over stdio transport
    mcp.run(transport="stdio")
