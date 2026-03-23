"""Build tennis court in Unity via MCP API."""
import requests
import json
import time

BASE_URL = "http://localhost:8080/mcp"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def init_session():
    resp = requests.post(BASE_URL, headers=HEADERS, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "court-builder", "version": "1.0"},
        },
    })
    sid = resp.headers.get("Mcp-Session-Id")
    print(f"Session: {sid}")
    return sid


def call_tool(session_id, name, args, call_id=2):
    h = {**HEADERS, "Mcp-Session-Id": session_id}
    resp = requests.post(BASE_URL, headers=h, json={
        "jsonrpc": "2.0", "id": call_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    })
    for line in resp.text.split("\n"):
        if line.startswith("data:"):
            return json.loads(line[5:])
    return {"raw": resp.text}


COURT_SCRIPT = r'''using UnityEngine;

[ExecuteInEditMode]
public class TennisCourtBuilder : MonoBehaviour
{
    const float CL = 23.77f;     // court length
    const float CW = 8.23f;      // court width (doubles)
    const float SO = 1.37f;      // singles offset
    const float NY = 11.885f;    // net Y
    const float SN = 5.485f;     // service line near
    const float SF = 18.285f;    // service line far
    const float NH = 1.07f;      // net height at posts
    const float NC = 0.914f;     // net center height
    const float LW = 0.05f;      // line width

    public void BuildCourt()
    {
        while (transform.childCount > 0)
            DestroyImmediate(transform.GetChild(0).gameObject);

        // Surround (green)
        var surr = GameObject.CreatePrimitive(PrimitiveType.Cube);
        surr.name = "Surround";
        surr.transform.SetParent(transform);
        surr.transform.localPosition = new Vector3(CW/2f, -0.05f, CL/2f);
        surr.transform.localScale = new Vector3(CW+10f, 0.1f, CL+10f);
        SetCol(surr, new Color(0.15f, 0.55f, 0.3f));

        // Court surface (terracotta)
        var court = GameObject.CreatePrimitive(PrimitiveType.Cube);
        court.name = "CourtSurface";
        court.transform.SetParent(transform);
        court.transform.localPosition = new Vector3(CW/2f, 0f, CL/2f);
        court.transform.localScale = new Vector3(CW, 0.01f, CL);
        SetCol(court, new Color(0.76f, 0.38f, 0.22f));

        // Lines
        var lp = new GameObject("Lines");
        lp.transform.SetParent(transform);
        L(lp,"BaseNear",0,0,CW,0);
        L(lp,"BaseFar",0,CL,CW,CL);
        L(lp,"SideL",0,0,0,CL);
        L(lp,"SideR",CW,0,CW,CL);
        L(lp,"SingleL",SO,0,SO,CL);
        L(lp,"SingleR",CW-SO,0,CW-SO,CL);
        L(lp,"SvcN",SO,SN,CW-SO,SN);
        L(lp,"SvcF",SO,SF,CW-SO,SF);
        L(lp,"CtrSvc",CW/2f,SN,CW/2f,SF);
        L(lp,"CtrN",CW/2f,0,CW/2f,0.15f);
        L(lp,"CtrF",CW/2f,CL,CW/2f,CL-0.15f);

        // Net
        var np = new GameObject("Net");
        np.transform.SetParent(transform);
        var nm = GameObject.CreatePrimitive(PrimitiveType.Cube);
        nm.name="NetMesh"; nm.transform.SetParent(np.transform);
        nm.transform.localPosition = new Vector3(CW/2f, NC/2f, NY);
        nm.transform.localScale = new Vector3(CW+1.8f, NC, 0.02f);
        SetCol(nm, new Color(0.9f,0.9f,0.9f));
        var pL = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        pL.name="PostL"; pL.transform.SetParent(np.transform);
        pL.transform.localPosition = new Vector3(-0.9f, NH/2f, NY);
        pL.transform.localScale = new Vector3(0.05f, NH/2f, 0.05f);
        var pR = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        pR.name="PostR"; pR.transform.SetParent(np.transform);
        pR.transform.localPosition = new Vector3(CW+0.9f, NH/2f, NY);
        pR.transform.localScale = new Vector3(0.05f, NH/2f, 0.05f);

        // Cameras (from real PnP calibration)
        var cp = new GameObject("Cameras");
        cp.transform.SetParent(transform);
        // cam66: world(4.321, -4.236, 7.0) -> Unity(4.321, 7.0, -4.236)
        var c66 = new GameObject("cam66");
        c66.transform.SetParent(cp.transform);
        c66.transform.localPosition = new Vector3(4.321f, 7.0f, -4.236f);
        var cam66 = c66.AddComponent<Camera>();
        cam66.fieldOfView = 2f*Mathf.Atan(1080f/(2f*1029.6f))*Mathf.Rad2Deg;
        c66.transform.LookAt(new Vector3(CW/2f, 0, CL/2f));
        cam66.enabled = false;
        // cam68: world(4.218, 28.908, 7.0) -> Unity(4.218, 7.0, 28.908)
        var c68 = new GameObject("cam68");
        c68.transform.SetParent(cp.transform);
        c68.transform.localPosition = new Vector3(4.218f, 7.0f, 28.908f);
        var cam68 = c68.AddComponent<Camera>();
        cam68.fieldOfView = 2f*Mathf.Atan(1080f/(2f*1261.6f))*Mathf.Rad2Deg;
        c68.transform.LookAt(new Vector3(CW/2f, 0, CL/2f));
        cam68.enabled = false;

        // Tennis ball
        var ball = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        ball.name = "TennisBall";
        ball.transform.SetParent(transform);
        ball.transform.localPosition = new Vector3(CW/2f, 1.5f, NY);
        ball.transform.localScale = Vector3.one * 0.067f;
        SetCol(ball, new Color(0.8f, 0.9f, 0.1f));
        var rb = ball.AddComponent<Rigidbody>();
        rb.mass = 0.057f; rb.drag = 0.5f;
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;
        var pm = new PhysicMaterial("BallPhys");
        pm.bounciness = 0.75f; pm.bounceCombine = PhysicMaterialCombine.Maximum;
        ball.GetComponent<Collider>().material = pm;

        Debug.Log("Tennis court built with cameras and ball!");
    }

    void L(GameObject p, string n, float x1, float z1, float x2, float z2)
    {
        var o = GameObject.CreatePrimitive(PrimitiveType.Cube);
        o.name=n; o.transform.SetParent(p.transform);
        float dx=x2-x1, dz=z2-z1;
        float len=Mathf.Sqrt(dx*dx+dz*dz);
        float ang=Mathf.Atan2(dx,dz)*Mathf.Rad2Deg;
        o.transform.localPosition = new Vector3((x1+x2)/2f, 0.006f, (z1+z2)/2f);
        o.transform.localRotation = Quaternion.Euler(0, ang, 0);
        o.transform.localScale = new Vector3(LW, 0.005f, len+LW);
        SetCol(o, Color.white);
    }

    void SetCol(GameObject o, Color c)
    {
        var m = new Material(Shader.Find("Standard"));
        m.color = c;
        o.GetComponent<Renderer>().material = m;
    }
}'''


def main():
    sid = init_session()
    cid = [2]

    def step(name, tool, args):
        cid[0] += 1
        print(f"\n=== {name} ===")
        result = call_tool(sid, tool, args, cid[0])
        # Extract text from result
        try:
            content = result.get("result", {}).get("content", [])
            for c in content:
                if c.get("type") == "text":
                    print(c["text"][:300])
        except Exception:
            print(json.dumps(result, indent=2)[:300])
        return result

    # Step 1: Create script
    step("Create script", "create_script", {
        "path": "Assets/Scripts/TennisCourtBuilder.cs",
        "contents": COURT_SCRIPT,
    })
    print("\nWaiting for compilation...")
    time.sleep(8)

    step("Check console", "read_console", {"count": 5})

    # Step 2: Create root object with component
    step("Create TennisCourt object", "manage_gameobject", {
        "action": "create",
        "name": "TennisCourt",
        "components_to_add": ["TennisCourtBuilder"],
    })

    # Step 3: Create light
    step("Create directional light", "manage_gameobject", {
        "action": "create",
        "name": "SunLight",
        "position": {"x": 4, "y": 15, "z": 12},
        "rotation": {"x": 50, "y": -30, "z": 0},
        "components_to_add": ["Light"],
    })

    # Step 4: Build the court using batch_execute with C# code
    step("Build court via execute", "batch_execute", {
        "commands": [
            {
                "tool": "manage_script_capabilities",
                "args": {
                    "action": "invoke_method",
                    "target_object": "TennisCourt",
                    "component_name": "TennisCourtBuilder",
                    "method_name": "BuildCourt",
                }
            }
        ]
    })

    # Alternatively try manage_editor to run in edit mode
    step("Check scene hierarchy", "manage_scene", {
        "action": "get_hierarchy",
    })

    print("\n=== Done! ===")
    print("If BuildCourt didn't auto-run, select TennisCourt in Unity hierarchy,")
    print("then right-click the TennisCourtBuilder component -> BuildCourt")


if __name__ == "__main__":
    main()
