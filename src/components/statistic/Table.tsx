`use-client`

import { ResponseBody } from '@/types/response-body';
import {
  Box,
  Button,
  Card,
  Center,
  Flex,
  Input,
  Select,
  SimpleGrid,
  Spinner,
  Table,
  TableContainer,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
  useColorModeValue,
} from '@chakra-ui/react';
import { useEffect, useState } from 'react';
import {useFetchAnalyticsTable, useFetchMasterWilayah} from '@/components/analytics/data'

export default function AnomalyTable() {

  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  // Filters
  const [kabFilter, setKabFilter] = useState('');
  const [kecFilter, setKecFilter] = useState('');
  const [desFilter, setDesFilter] = useState('');

//   options
  const [kabOptions, setKabOptions] = useState([]);
  const [kecOptions, setKecOptions] = useState([]);
  const [desOptions, setDesOptions] = useState([]);

const fetchRows = async () => {
    setLoading(true);
    const params = new URLSearchParams({
      page: String(page),
      limit: '5',
      kabupaten: kabFilter,
      kecamatan: kecFilter,
      desa: desFilter,
    });
    try {
        let response = await useFetchAnalyticsTable(params);
        if (response.success) {
            const data = response.data;
            setRows(data.rows);
            setTotalPages(data.totalPages);
            setLoading(false);
        } else {
            throw Error(response.message);
        }
    } catch(err: any) {
        console.log(err)
    }
    setLoading(false);
  };

  const fetchMasterWilayah = async (level:string) => {
    let params = new URLSearchParams({
        level: level
    })
    try {
        let response = await useFetchMasterWilayah(params);
        if (response.success) {
            if (level == 'kab') {
                setKabOptions(response.data);
            } else if (level == 'kec') {
                setKecOptions(response.data);
            } else if (level == 'des') {
                setDesOptions(response.data);
            }
        } else {
            throw Error(response.message);
        }
    } catch(err: any) {
        console.log(err)
    }
  }

  useEffect(()=> {
    fetchMasterWilayah('kab');
    fetchMasterWilayah('kec');
    fetchMasterWilayah('des');
    fetchRows();
  }, [page])
  
  const handleFilter = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchRows();
  };

  return (
    <Box p={0}>
      <form onSubmit={handleFilter}>
        <Flex gap="2" mb="4" wrap="wrap" direction="row">
                <Select placeholder="Pilih Kabupaten" value={kabFilter} onChange={(e) => setKabFilter(e.target.value)} size='sm' w={'32%'} disabled={kabOptions.length == 0}>
                    {kabOptions.map((elem: any) => (
                        <option value={elem.idkab}>{elem.nmkab}</option>
                    ))}
                </Select>
                <Select placeholder="Pilih Kecamatan" value={kabFilter} onChange={(e) => setKecFilter(e.target.value)} size='sm' w={'32%'} disabled={kecOptions.length == 0}>
                    {kecOptions.map((elem: any) => (
                        <option value={elem.idkec}>{elem.nmkec}</option>
                    ))}
                </Select>
                <Select placeholder="Pilih Desa" value={kabFilter} onChange={(e) => setDesFilter(e.target.value)} size='sm' w={'32%'} disabled={desOptions.length == 0}>
                    {desOptions.map((elem: any) => (
                        <option value={elem.iddesa}>{elem.nmdesa}</option>
                    ))}
                </Select>
        </Flex>
          <Button type="submit" colorScheme="blue" size='sm'>Apply Filters</Button>
      </form>

      {loading ? (
        <Center w={'100%'}>

        <Spinner />
        </Center>
      ) : (
        <TableContainer w={'100%'}>
        <Table variant="striped" size="sm">
          <Thead>
            <Tr>
              <Th>Anomaly Label</Th>
              <Th isNumeric>Anomaly Proba</Th>
              <Th isTruncated>Desa</Th>
              <Th isTruncated>Kecamatan</Th>
              <Th isTruncated>Kabupaten</Th>
            </Tr>
          </Thead>
          <Tbody>
            {rows.map((r: any) => (
              <Tr key={r.id}>
                <Td>{r.anomaly_label}</Td>
                <Td isNumeric>{parseFloat(r.anomaly_score_probability.toFixed(2))}</Td>
                <Td isTruncated>{r.nmdesa}</Td>
                <Td isTruncated>{r.nmkec}</Td>
                <Td isTruncated>{r.nmkab}</Td>
              </Tr>
            ))}
          </Tbody>
        </Table></TableContainer>
      )}

      <Flex mt="4" gap="2" align="center">
        <Button onClick={() => setPage(page - 1)} disabled={page === 1}>Prev</Button>
        <span>Page {page} of {totalPages}</span>
        <Button onClick={() => setPage(page + 1)} disabled={page === totalPages}>Next</Button>
      </Flex>
    </Box>
  );
}