import { NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { getResponseBody, ResponseBody } from '@/types/response-body';

export async function GET(req: Request) {
  const response: ResponseBody = getResponseBody();
  try {
    const { searchParams } = new URL(req.url);
    // table
    const level = searchParams.get('level') || '';
    const idkab = searchParams.get('idkab') || '';
    const idkec = searchParams.get('idkec') || '';
    let params: any[] = [];
    let idx = 1;

    let cols: string = [];
    if (level == 'kab') {
      cols = 'idkab, nmkab';
    } else if (level == 'kec') {
      cols = 'idkec, nmkec';
    } else if (level == 'des') {
      cols = 'iddesa, nmdesa';
    }
    let query = `SELECT distinct ${cols} FROM table_h3_anomaly_score WHERE 1=1`;

    if (idkab) {
      query += ` AND idkab = $${idx++}`;
      params.push(`%${idkab}%`);
    }

    if (idkec) {
      query += ` AND idkec = $${idx++}`;
      params.push(`%${idkec}%`);
    }

    const { rows } = await pool.query(query, params);

    // Count total
    response.data = rows;

    return NextResponse.json(response);
  } catch (err: any) {
    response.success = false;
    response.message = err.message || 'Terjadi kesalahan!';
    return NextResponse.json(response);
  }
}
