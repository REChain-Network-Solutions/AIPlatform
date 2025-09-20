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

abstract class AbstractComparisonValidator  extends ConstraintValidator
{
    /**
     * @throws InvalidPropertyValueException
     */
    public function validate(mixed $value, AbstractComparison $constraint): void
    {
        if (null === $value) {
            return;
        }

        $comparedValue = $constraint->getValue();

        if (\is_string($comparedValue) && $value instanceof \DateTimeInterface) {
            $dateTimeClass = $value instanceof \DateTimeImmutable ? \DateTimeImmutable::class : \DateTime::class;

            $comparedValue = new $dateTimeClass($comparedValue);
        }

        if (!$this->compareValues($value, $comparedValue)) {
            throw new InvalidPropertyValueException(
                $this->generateMessage(self::INVALID_PROP_VALUE_TMPL, $constraint->getMessage()),
                0,
                $this->getPropClassConcat(),
                $value
            );
        }
    }

    abstract protected function compareValues(mixed $value1, mixed $value2): bool;
}
