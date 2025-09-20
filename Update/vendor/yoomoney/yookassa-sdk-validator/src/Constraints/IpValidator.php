<?php

/**
 * The MIT License
 *
 * Copyright (c) 2024 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Validator\Constraints;

use YooKassa\Validator\Exceptions\InvalidPropertyValueException;
use YooKassa\Validator\Exceptions\ValidatorParameterException;

class IpValidator extends ConstraintValidator
{
    /**
     * @throws InvalidPropertyValueException
     */
    public function validate(mixed $value, Ip $constraint): void
    {
        if (null === $value || '' === $value) {
            return;
        }

        if (!\is_scalar($value) && !$value instanceof \Stringable) {
            throw new ValidatorParameterException('Value must be scalar or type of \Stringable');
        }

        $value = (string) $value;

        $flag = match ($constraint->getVersion()) {
            Ip::V4 => \FILTER_FLAG_IPV4,
            Ip::V6 => \FILTER_FLAG_IPV6,
            Ip::V4_NO_PRIV => \FILTER_FLAG_IPV4 | \FILTER_FLAG_NO_PRIV_RANGE,
            Ip::V6_NO_PRIV => \FILTER_FLAG_IPV6 | \FILTER_FLAG_NO_PRIV_RANGE,
            Ip::ALL_NO_PRIV => \FILTER_FLAG_NO_PRIV_RANGE,
            Ip::V4_NO_RES => \FILTER_FLAG_IPV4 | \FILTER_FLAG_NO_RES_RANGE,
            Ip::V6_NO_RES => \FILTER_FLAG_IPV6 | \FILTER_FLAG_NO_RES_RANGE,
            Ip::ALL_NO_RES => \FILTER_FLAG_NO_RES_RANGE,
            Ip::V4_ONLY_PUBLIC => \FILTER_FLAG_IPV4 | \FILTER_FLAG_NO_PRIV_RANGE | \FILTER_FLAG_NO_RES_RANGE,
            Ip::V6_ONLY_PUBLIC => \FILTER_FLAG_IPV6 | \FILTER_FLAG_NO_PRIV_RANGE | \FILTER_FLAG_NO_RES_RANGE,
            Ip::ALL_ONLY_PUBLIC => \FILTER_FLAG_NO_PRIV_RANGE | \FILTER_FLAG_NO_RES_RANGE,
            default => 0,
        };

        if (!filter_var($value, \FILTER_VALIDATE_IP, $flag)) {
            throw new InvalidPropertyValueException(
                $this->generateMessage(self::INVALID_PROP_VALUE_TMPL, $constraint->getMessage()),
                0,
                $this->getPropClassConcat(),
                $value
            );
        }
    }
}
