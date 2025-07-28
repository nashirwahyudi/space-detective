import { NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { getResponseBody, ResponseBody } from '@/types/response-body';

export async function GET(req: Request) {
  const response: ResponseBody = getResponseBody();
  try {
    const { searchParams } = new URL(req.url);
    // paging
    const page = parseInt(searchParams.get('page') || '1', 10);
    const limit = parseInt(searchParams.get('limit') || '10', 10);
    const offset = (page - 1) * limit;
    // filters
    const kabupaten = searchParams.get('kabupaten') || '';
    const kecamatan = searchParams.get('kecamatan') || '';
    const desa = searchParams.get('desa') || '';
    let params: any[] = [];
    let query = `SELECT * FROM table_h3_anomaly_score WHERE 1=1`;
    let idx = 1;
    if (kabupaten) {
      query += ` AND nmkab ILIKE $${idx++}`;
      params.push(`%${kabupaten}%`);
    }
    if (kecamatan) {
      query += ` AND nmkec ILIKE $${idx++}`;
      params.push(`%${kecamatan}%`);
    }
    if (desa) {
      query += ` AND nmdes ILIKE $${idx++}`;
      params.push(`%${desa}%`);
    }

    query += ` ORDER BY anomaly_flag DESC LIMIT $${idx++} OFFSET $${idx++}`;
    params.push(limit, offset);
    const { rows } = await pool.query(query, params);

    // Count total
    let countQuery = `SELECT COUNT(*) FROM table_h3_anomaly_score WHERE 1=1`;
    const countParams: any[] = [];
    let cIdx = 1;

    if (kabupaten) {
      countQuery += ` AND nmkab ILIKE $${cIdx++}`;
      countParams.push(`%${kabupaten}%`);
    }
    if (kecamatan) {
      countQuery += ` AND nmkec ILIKE $${cIdx++}`;
      countParams.push(`%${kecamatan}%`);
    }
    if (desa) {
      countQuery += ` AND nmdes ILIKE $${cIdx++}`;
      countParams.push(`%${desa}%`);
    }

    const totalResult = await pool.query(countQuery, countParams);

    response.data = {
      rows: rows,
      totalResult: totalResult,
      totalPages: Math.ceil(totalResult.rows[0].count / limit),
    };

    return NextResponse.json(response);
  } catch (err: any) {
    response.success = false;
    response.message = err.message || 'Terjadi kesalahan!';
    return NextResponse.json(response);
  }
}
