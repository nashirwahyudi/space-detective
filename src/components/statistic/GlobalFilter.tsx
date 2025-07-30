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

export default function GlobalFilter(props: {
  onFilterSubmit: (iddesa: string, idkec: string, idkab: string) => void;
}) {
  // Filters
  const [kabFilter, setKabFilter] = useState('');
  const [kecFilter, setKecFilter] = useState('');
  const [desFilter, setDesFilter] = useState('');

  //   options
  const [kabOptions, setKabOptions] = useState([]);
  const [kecOptions, setKecOptions] = useState([]);
  const [desOptions, setDesOptions] = useState([]);

  const fetchMasterWilayah = async (level: string) => {
    let params = new URLSearchParams({
      level: level,
    });
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
    } catch (err: any) {
      console.log(err);
    }
  };

  useEffect(() => {
    fetchMasterWilayah('kab');
    fetchMasterWilayah('kec');
    fetchMasterWilayah('des');
  }, []);

  const handleFilter = (e: React.FormEvent) => {
    e.preventDefault();
    props.onFilterSubmit(desFilter, kecFilter, kabFilter);
  };
  return (
    <form onSubmit={handleFilter}>
      <Flex gap="2" mb="4" wrap="wrap" direction="row">
        <Select
          placeholder="Pilih Kabupaten"
          value={kabFilter}
          onChange={(e) => setKabFilter(e.target.value)}
          size="sm"
          w={'32%'}
          disabled={kabOptions.length == 0}
        >
          {kabOptions.map((elem: any) => (
            <option value={elem.idkab} key={elem.idkab}>
              {elem.nmkab}
            </option>
          ))}
        </Select>
        <Select
          placeholder="Pilih Kecamatan"
          value={kecFilter}
          onChange={(e) => setKecFilter(e.target.value)}
          size="sm"
          w={'32%'}
          disabled={kecOptions.length == 0}
        >
          {kecOptions.map((elem: any) => (
            <option value={elem.idkec} key={elem.idkec}>
              {elem.nmkec}
            </option>
          ))}
        </Select>
        <Select
          placeholder="Pilih Desa"
          value={desFilter}
          onChange={(e) => setDesFilter(e.target.value)}
          size="sm"
          w={'32%'}
          disabled={desOptions.length == 0}
        >
          {desOptions.map((elem: any) => (
            <option value={elem.iddesa} key={elem.iddesa}>
              {elem.nmdesa}
            </option>
          ))}
        </Select>
      </Flex>
      <Button type="submit" colorScheme="blue" size="sm">
        Apply Filters
      </Button>
    </form>
  );
}
