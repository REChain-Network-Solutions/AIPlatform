<?php

namespace Tests\YooKassa\Common\Exceptions;

use DateTime;
use stdClass;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;

/**
 * @internal
 */
class InvalidPropertyValueExceptionTest extends InvalidPropertyExceptionTest
{
    /**
     * @dataProvider validValueDataProvider
     *
     * @param mixed $value
     */
    public function testGetValue($value): void
    {
        $instance = $this->getTestInstance('', '', $value);
        if (null !== $value) {
            self::assertEquals($value, $instance->getValue());
        } else {
            self::assertNull($instance->getValue());
        }
    }

    public static function validValueDataProvider()
    {
        return [
            [null],
            [''],
            ['value'],
            [['test']],
            [new stdClass()],
            [new DateTime()],
        ];
    }

    protected function getTestInstance($message, $property, $value = null): InvalidPropertyValueException
    {
        return new InvalidPropertyValueException($message, 0, $property, $value);
    }
}
