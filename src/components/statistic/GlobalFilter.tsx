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
  useFecthShap,
  useFetchAnalyticsTable,
  useFetchMasterWilayah,
} from '@/components/analytics/data';

export default function GlobalFilter(props: {
  onFilterSubmit: (
    h3_index: string,
    iddesa: string,
    idkec: string,
    idkab: string,
  ) => void;
}) {
  // Filters
  const [kabFilter, setKabFilter] = useState('');
  const [kecFilter, setKecFilter] = useState('');
  const [desFilter, setDesFilter] = useState('');
  const [h3IndexFilter, setH3IndexFilter] = useState('');

  //   options
  const [kabOptions, setKabOptions] = useState([]);
  const [kecOptions, setKecOptions] = useState([]);
  const [desOptions, setDesOptions] = useState([]);
  const [h3IndexOptions, setS3IndexOptions] = useState([]);

  const fetchMasterWilayah = async (
    level: string,
    idkab: string | '',
    idkec: string | '',
    iddesa: string | '',
  ) => {
    let params = new URLSearchParams({
      level: level,
      idkab: idkab,
      idkec: idkec,
      iddesa: iddesa,
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
        } else if (level == 'h3') {
          setS3IndexOptions(response.data);
        }
      } else {
        throw Error(response.message);
      }
    } catch (err: any) {
      console.log(err);
    }
  };

  const fetchShaps = async (
    idkab: string | '',
    idkec: string | '',
    iddesa: string | '',
  ) => {
    let params = new URLSearchParams({
      idkab: idkab,
      idkec: idkec,
      iddesa: iddesa,
    });
    try {
      let response = await useFecthShap(params);
      if (response.success) {
        setS3IndexOptions(response.data);
        setH3IndexFilter(response.data[0].png_relative_path);
      } else {
        throw Error(response.message);
      }
    } catch (err: any) {
      console.log(err);
    }
  };

  useEffect(() => {
    fetchMasterWilayah('kab', '', '', '');
    fetchMasterWilayah('kec', kabFilter, '', '');
    fetchMasterWilayah('des', kabFilter, kecFilter, '');
    fetchMasterWilayah('des', kabFilter, kecFilter, desFilter);
  }, []);

  useEffect(() => {
    if (kabFilter != '') {
      if (kecFilter == '') {
        fetchMasterWilayah('kec', kabFilter, '', '');
      }
      if (desFilter == '') {
        fetchMasterWilayah('des', kabFilter, kecFilter, '');
      }
      fetchMasterWilayah('h3', kabFilter, kecFilter, desFilter);
    }
  }, [kabFilter, kecFilter, desFilter]);

  // useEffect(() => {
  //   fetchShaps(kabFilter, kecFilter, desFilter)
  // },[kabFilter, kecFilter, desFilter])

  const handleFilter = (e: React.FormEvent) => {
    e.preventDefault();
    props.onFilterSubmit(h3IndexFilter, desFilter, kecFilter, kabFilter);
  };
  return (
    <form onSubmit={handleFilter}>
      <Flex gap="2" mb="4" wrap="wrap" direction="row">
        <Select
          placeholder="Pilih Kabupaten"
          value={kabFilter}
          onChange={(e) => setKabFilter(e.target.value)}
          size="sm"
          w={{ base: '33%', md: '33%', sm: '30%' }}
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
          w={{ base: '32.3%', md: '32.2%', sm: '30%' }}
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
          w={{ base: '33%', md: '33%', sm: '30%' }}
          disabled={desOptions.length == 0}
        >
          {desOptions.map((elem: any) => (
            <option value={elem.iddesa} key={elem.iddesa}>
              {elem.nmdesa}
            </option>
          ))}
        </Select>

        <Select
          placeholder="Pilih Index H3"
          // value={h3IndexFilter}
          onChange={(e) => setH3IndexFilter(e.target.value)}
          size="sm"
          w={{ base: '100%', lg: '100%', md: '33%' }}
          disabled={h3IndexOptions.length == 0}
          defaultValue={h3IndexFilter}
        >
          {h3IndexOptions.map((elem: any, index: any) => (
            <option value={elem.h3_index} key={`${elem.h3_index}_${index}`}>
              {elem.h3_index} - {elem.anomaly_score_probability?.toFixed(2)}
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
