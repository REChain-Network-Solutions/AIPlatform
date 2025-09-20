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

class TimeValidator extends ConstraintValidator
{
    public const PATTERN = '/^(\d{2}):(\d{2}):(\d{2})$/';

    public static function checkTime(int $hour, int $minute, float $second): bool
    {
        return $hour >= 0 && $hour < 24 && $minute >= 0 && $minute < 60 && $second >= 0 && $second < 60;
    }

    /**
     * @throws InvalidPropertyValueException|ValidatorParameterException
     */
    public function validate(mixed $value, Time $constraint): void
    {
        if (null === $value || '' === $value || $value instanceof \DateTime) {
            return;
        }

        if (!\is_scalar($value) && !$value instanceof \Stringable) {
            throw new ValidatorParameterException('Value must be scalar or type of \Stringable');
        }

        $value = (string) $value;

        if (!preg_match(static::PATTERN, $value, $matches)) {
            throw new InvalidPropertyValueException(
                $this->generateMessage(self::INVALID_PROP_VALUE_TMPL, $constraint->getMessage()),
                0,
                $this->getPropClassConcat(),
                $value
            );
        }

        if (!self::checkTime($matches[1], $matches[2], $matches[3])) {
            throw new InvalidPropertyValueException(
                $this->generateMessage(self::INVALID_PROP_VALUE_TMPL, $constraint->getMessage()),
                0,
                $this->getPropClassConcat(),
                $value
            );
        }
    }
}
