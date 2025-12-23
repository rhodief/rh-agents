#!/usr/bin/env python3
"""
Simple client to test the streaming API.
"""
import requests
import json
import sys

def test_streaming_api(port=8001, query=None):
    """Test the streaming API endpoint."""
    
    if query is None:
        query = "Faça um relatório com o resumo dos óbices jurídicos da decisão de Admissibilidade."
    
    url = f"http://localhost:{port}/api/stream"
    
    payload = {
        "query": query,
        "use_cache": True
    }
    
    print(f"🚀 Testing Streaming API at {url}")
    print(f"📝 Query: {query}")
    print("=" * 80)
    print()
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            
            print("✅ Connected! Streaming events...")
            print("=" * 80)
            print()
            
            event_count = 0
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data_str = decoded_line[6:]  # Remove 'data: ' prefix
                        try:
                            event_data = json.loads(data_str)
                            event_count += 1
                            
                            # Format and print the event
                            event_type = event_data.get('event_type', 'unknown')
                            timestamp = event_data.get('timestamp', '')
                            
                            print(f"[{event_count}] {event_type.upper()}")
                            
                            if timestamp:
                                print(f"    ⏰ Time: {timestamp}")
                            
                            if 'actor_name' in event_data and event_data['actor_name']:
                                print(f"    🎭 Actor: {event_data['actor_name']}")
                            
                            if 'detail' in event_data and event_data['detail']:
                                detail = event_data['detail']
                                if len(detail) > 200:
                                    detail = detail[:200] + "..."
                                print(f"    📄 Detail: {detail}")
                            
                            if 'message' in event_data and event_data['message']:
                                print(f"    💬 Message: {event_data['message']}")
                            
                            if event_data.get('cached'):
                                print(f"    ⚡ [CACHED]")
                            
                            if 'error' in event_data:
                                print(f"    ❌ Error: {event_data['error']}")
                            
                            if 'result' in event_data:
                                result = event_data['result']
                                if len(result) > 500:
                                    result = result[:500] + "..."
                                print(f"    ✨ Result: {result}")
                            
                            if 'cache_stats' in event_data:
                                stats = event_data['cache_stats']
                                print(f"    📊 Cache Stats:")
                                print(f"       - Backend: {stats.get('backend', 'N/A')}")
                                print(f"       - Entries: {stats.get('size', 0)}")
                                print(f"       - Hits: {stats.get('hits', 0)}")
                                print(f"       - Misses: {stats.get('misses', 0)}")
                                print(f"       - Hit Rate: {stats.get('hit_rate', 0)*100:.1f}%")
                            
                            print()
                            
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Failed to parse event: {e}")
                            print(f"    Raw data: {data_str[:100]}")
                            print()
            
            print("=" * 80)
            print(f"✅ Stream completed! Total events received: {event_count}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to {url}")
        print(f"   Make sure the server is running on port {port}")
        print(f"   Start it with: python streaming_api.py --port {port}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"❌ Error: Request timed out")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Stream interrupted by user")
        sys.exit(0)


def check_server_health(port=8001):
    """Check if the server is running."""
    url = f"http://localhost:{port}/health"
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Server is healthy: {data}")
        return True
    except:
        print(f"❌ Server is not responding on port {port}")
        return False


if __name__ == "__main__":
    port = 8001
    query = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:]):
            if arg == '--port' and i + 2 < len(sys.argv):
                try:
                    port = int(sys.argv[i + 2])
                except ValueError:
                    print(f"Invalid port: {sys.argv[i + 2]}")
                    sys.exit(1)
            elif arg == '--query' and i + 2 < len(sys.argv):
                query = sys.argv[i + 2]
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║          RH AGENTS STREAMING API TEST CLIENT                 ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Check server health first
    if not check_server_health(port):
        print(f"\n💡 Start the server first:")
        print(f"   python streaming_api.py --port {port}")
        sys.exit(1)
    
    print()
    
    # Test the streaming
    test_streaming_api(port, query)
