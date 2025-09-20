<?php

namespace Tests\YooKassa\Common\Exceptions;

use DateTime;
use stdClass;
use YooKassa\Common\Exceptions\InvalidPropertyException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;

/**
 * @internal
 */
class InvalidPropertyValueTypeExceptionTest extends InvalidPropertyExceptionTest
{
    /**
     * @dataProvider validTypeDataProvider
     *
     * @param mixed $value
     */
    public function testGetType($value, string $type): void
    {
        $instance = $this->getTestInstance('', '', $value);
        self::assertEquals($type, $instance->getType());
    }

    public static function validTypeDataProvider()
    {
        return [
            [null, 'null'],
            ['', 'string'],
            ['value', 'string'],
            [['test'], 'array'],
            [new stdClass(), 'stdClass'],
            [new DateTime(), 'DateTime'],
            [new InvalidPropertyException(), 'YooKassa\\Common\\Exceptions\\InvalidPropertyException'],
            [fopen(__FILE__, 'rb'), 'resource'],
            [true, 'boolean'],
            [false, 'boolean'],
            [0, 'integer'],
            [0.01, 'double'],
        ];
    }

    /**
     * @param null $value
     */
    protected function getTestInstance(string $message, string $property, $value = null): InvalidPropertyValueTypeException
    {
        return new InvalidPropertyValueTypeException($message, 0, $property, $value);
    }
}
