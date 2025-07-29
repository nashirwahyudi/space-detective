import { pool } from '@/lib/db';
import { ResponseBody, getResponseBody } from '@/types/response-body';
import { NextResponse } from 'next/server';

export async function GET(req: Request) {
  const response: ResponseBody = getResponseBody();
  try {
    const { searchParams } = new URL(req.url);
    // filters
    const idkab = searchParams.get('idkab') || '';
    const idkec = searchParams.get('idkec') || '';
    const iddesa = searchParams.get('iddesa') || '';
    let params: any[] = [];
    let query = `SELECT h3_index, iddesa, idkec, idkab, nmdesa, nmkec, nmkab, ST_AsGeoJSON(geometry)::json as geometry, anomaly_label, anomaly_score_probability FROM map_sumut_with_all_feature WHERE 1=1`;

    let idx = 1;
    if (idkab) {
      query += ` AND idkab = $${idx++}`;
      params.push(idkab);
    }
    if (idkec) {
      query += ` AND idkec = $${idx++}`;
      params.push(idkec);
    }
    if (iddesa) {
      query += ` AND iddes = $${idx++}`;
      params.push(iddesa);
    }
    const res = await pool.query(query, params);
    const geojson = {
      type: 'FeatureCollection',
      features: res.rows.map((row: any) => {
        return {
          type: 'Feature',
          geometry: row.geometry,
          properties: {
            iddesa: row.iddesa,
            idkab: row.name,
            idkec: row.idkec,
            nmdesa: row.nmdesa,
            nmkab: row.nmkab,
            nmkec: row.nmkec,
            anomaly_score_probability: row.anomaly_score_probability,
            anomaly_label: row.anomaly_label,
          },
        };
      }),
    };
    response.data = geojson;
    return NextResponse.json(response);
  } catch (err: any) {
    response.success = false;
    response.message = err.message || 'Terjadi kesalahan!';
    return NextResponse.json(response);
  }
}
