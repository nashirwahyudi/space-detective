import { Box, Flex, Text } from '@chakra-ui/react';

export const MapLegend = () => (
  <Box
    position="absolute"
    bottom={4}
    left={4}
    bg="white"
    p={3}
    borderRadius="md"
    boxShadow="md"
    zIndex={10}
  >
    <Text fontWeight="bold" mb={2}>
      Probability
    </Text>
    <Flex align="center" gap={2}>
      <Box w="20px" h="20px" bg="#16A34A" />{' '}
      <Text fontSize="xs">Very Low (0.0–0.2)</Text>
    </Flex>
    <Flex align="center" gap={2}>
      <Box w="20px" h="20px" bg="#84CC16" />{' '}
      <Text fontSize="xs">Low (0.2–0.4)</Text>
    </Flex>
    <Flex align="center" gap={2}>
      <Box w="20px" h="20px" bg="#FACC15" />{' '}
      <Text fontSize="xs">Moderate (0.4–0.6)</Text>
    </Flex>
    <Flex align="center" gap={2}>
      <Box w="20px" h="20px" bg="#FB923C" />{' '}
      <Text fontSize="xs">High (0.6–0.8)</Text>
    </Flex>
    <Flex align="center" gap={2}>
      <Box w="20px" h="20px" bg="#DC2626" />{' '}
      <Text fontSize="xs">Very High (0.8–1.0)</Text>
    </Flex>
  </Box>
);
