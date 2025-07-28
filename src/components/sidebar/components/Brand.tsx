'use client';
// Chakra imports
import { Flex, useColorModeValue, Text } from '@chakra-ui/react';
import { HorizonLogo } from '@/components/icons/Icons';
import { HSeparator } from '@/components/separator/Separator';

export function SidebarBrand() {
  //   Chakra color mode
  let logoColor = useColorModeValue('navy.700', 'white');

  return (
    <Flex alignItems="center" flexDirection="column">
      {/* <HorizonLogo h="26px" w="146px" my="30px" color={logoColor} /> */}
      <Flex align="center" flexDirection="row" my="15px">
        <Text
          color="#000"
          fontWeight="700"
          fontSize="xl"
          as="span"
          marginEnd="2"
        >
          SPACE
        </Text>{' '}
        <Text color="#000" fontWeight="500" fontSize="xl" as="span">
          DETECTIVE
        </Text>
      </Flex>
      <HSeparator mb="20px" w="284px" />
    </Flex>
  );
}

export default SidebarBrand;
