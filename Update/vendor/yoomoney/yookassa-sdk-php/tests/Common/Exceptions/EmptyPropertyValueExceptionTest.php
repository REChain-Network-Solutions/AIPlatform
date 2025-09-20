<?php

namespace Tests\YooKassa\Common\Exceptions;

use YooKassa\Common\Exceptions\EmptyPropertyValueException;

/**
 * @internal
 */
class EmptyPropertyValueExceptionTest extends InvalidPropertyExceptionTest
{
    protected function getTestInstance(string $message, string $property): EmptyPropertyValueException
    {
        return new EmptyPropertyValueException($message, 0, $property);
    }
}
