import { mergeBufferAttributes, mergeBufferGeometries } from "three-stdlib";

function computeTangents() {
  throw new Error("BufferGeometryUtils: computeTangents renamed to computeMikkTSpaceTangents.");
}

function normalizeCollection(geometries) {
  if (!Array.isArray(geometries)) return [];
  return geometries.filter(Boolean);
}

export function mergeGeometries(geometries, useGroups = false) {
  const collection = normalizeCollection(geometries);
  if (!collection.length) return null;
  return mergeBufferGeometries(collection, useGroups);
}

export { mergeBufferAttributes, mergeBufferGeometries };
export { computeTangents };

const utils = {
  computeTangents,
  mergeBufferAttributes,
  mergeBufferGeometries,
  mergeGeometries,
};

export default utils;

