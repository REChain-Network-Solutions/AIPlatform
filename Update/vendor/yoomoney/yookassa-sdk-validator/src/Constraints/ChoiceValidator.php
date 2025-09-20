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

class ChoiceValidator extends ConstraintValidator
{
    /**
     * @throws InvalidPropertyValueException|ValidatorParameterException
     */
    public function validate(mixed $value, Choice $constraint): void
    {
        if ($constraint->isMultiple() && !\is_array($value)) {
            throw new ValidatorParameterException('Value must be type of array.');
        }

        if (null === $value) {
            return;
        }

        if ($constraint->getCallback()) {
            if (!\is_callable($choices = $constraint->getCallback())) {
                throw new ValidatorParameterException('The Choice constraint expects a valid callback.');
            }
            $choices = $choices();
        } else {
            $choices = $constraint->getChoices();
        }

        if ($constraint->isMultiple()) {
            foreach ($value as $valueItem) {
                if ($constraint->isMatch() xor \in_array($valueItem, $choices, true)) {
                    throw new InvalidPropertyValueException(
                        $this->generateMessage(self::INVALID_PROP_VALUE_TMPL, $constraint->getMessage()),
                        0,
                        $this->getPropClassConcat(),
                        $value
                    );
                }
            }
        } elseif ($constraint->isMatch() xor \in_array($value, $choices)) {
            throw new InvalidPropertyValueException(
                $this->generateMessage(self::INVALID_PROP_VALUE_TMPL, $constraint->getMultipleMessage()),
                0,
                $this->getPropClassConcat(),
                $value
            );
        }
    }
}
