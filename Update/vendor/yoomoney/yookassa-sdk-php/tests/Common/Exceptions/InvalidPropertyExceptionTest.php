<?php

namespace Tests\YooKassa\Common\Exceptions;

use PHPUnit\Framework\TestCase;
use YooKassa\Common\Exceptions\InvalidPropertyException;
use YooKassa\Helpers\StringObject;

/**
 * @internal
 */
class InvalidPropertyExceptionTest extends TestCase
{
    /**
     * @dataProvider validPropertyDataProvider
     *
     * @param mixed $property
     */
    public function testGetProperty($property): void
    {
        $instance = $this->getTestInstance('', $property);
        self::assertEquals((string) $property, $instance->getProperty());
    }

    public static function validPropertyDataProvider()
    {
        return [
            [''],
            ['property'],
            [new StringObject('property')],
        ];
    }

    protected function getTestInstance(string $message, string $property): InvalidPropertyException
    {
        return new InvalidPropertyException($message, 0, $property);
    }
}
