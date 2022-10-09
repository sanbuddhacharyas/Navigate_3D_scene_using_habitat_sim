import path from "path";
import fs from "fs";
import generateNavMeshJSON from "gltf-navmesh-generator";

// Works with .gltf and .glb
// .glb input preferred.
const navMeshGltfPath = path.join(__dirname, "navmesh.gltf");

// Where to output navmesh file
const navMeshJSONPath = path.join(__dirname, "navmesh.json");

// The name of the Object3D containing the navmesh geometry
const navMeshObjName = "Navmesh";

// Generates navmesh data that can be loaded in aframe via <a-entity nav-mesh="src: navmesh.json">
const outNavMeshPath = await generateNavMeshJSON(navMeshGltfPath, navMeshJSONPath, navMeshObjName);