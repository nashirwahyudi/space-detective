`use-client`;

import {
  Box,
  Button,
  Center,
  Flex,
  Select,
  Spinner,
  Table,
  TableContainer,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from '@chakra-ui/react';
import { useEffect, useState } from 'react';
import {
  useFetchAnalyticsTable,
  useFetchMasterWilayah,
} from '@/components/analytics/data';

export default function AnomalyTable(props: {
  iddesa: string;
  idkec: string;
  idkab: string;
}) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const fetchRows = async () => {
    setLoading(true);
    const params = new URLSearchParams({
      page: String(page),
      limit: '5',
      kabupaten: props.idkab,
      kecamatan: props.idkec,
      desa: props.iddesa,
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
    } catch (err: any) {
      console.log(err);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchRows();
  }, [page, props]);

  const handleFilter = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchRows();
  };

  return (
    <Box p={0}>
      {loading ? (
        <Center w={'100%'} h={'50%  '}>
          <Spinner />
        </Center>
      ) : (
        <TableContainer w={'100%'}>
          <Table variant="striped" size="sm">
            <Thead>
              <Tr key="table-1">
                <Th>Anomaly Label</Th>
                <Th isNumeric>Anomaly Proba</Th>
                <Th isTruncated>Desa</Th>
                <Th isTruncated>Kecamatan</Th>
                <Th isTruncated>Kabupaten</Th>
              </Tr>
            </Thead>
            <Tbody>
              {rows.map((r: any, index) => (
                <Tr key={`table_data_${index}`}>
                  <Td>{r.anomaly_label}</Td>
                  <Td isNumeric>
                    {parseFloat(r.anomaly_score_probability.toFixed(2))}
                  </Td>
                  <Td isTruncated>{r.nmdesa}</Td>
                  <Td isTruncated>{r.nmkec}</Td>
                  <Td isTruncated>{r.nmkab}</Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </TableContainer>
      )}

      <Flex mt="4" gap="2" align="center">
        <Button onClick={() => setPage(page - 1)} disabled={page === 1}>
          Prev
        </Button>
        <span>
          Page {page} of {totalPages}
        </span>
        <Button
          onClick={() => setPage(page + 1)}
          disabled={page === totalPages}
        >
          Next
        </Button>
      </Flex>
    </Box>
  );
}
