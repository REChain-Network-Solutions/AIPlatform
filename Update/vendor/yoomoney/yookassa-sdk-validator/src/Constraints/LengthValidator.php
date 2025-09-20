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

class LengthValidator extends ConstraintValidator
{
    /**
     * @throws InvalidPropertyValueException|ValidatorParameterException
     */
    public function validate(mixed $value, Length $constraint): void
    {
        if (null === $value) {
            return;
        }

        $invalidCharset = !@mb_check_encoding($value, $constraint->charset);

        if ($invalidCharset) {
            throw new ValidatorParameterException($constraint->getCharsetMessage());
        }

        $length = mb_strlen($value, $constraint->getCharset());

        if (null !== $constraint->getMax() && $length > $constraint->getMax()) {
            $exactlyOptionEnabled = $constraint->getMin() == $constraint->getMax();
            $constraintMessage = $exactlyOptionEnabled ? $constraint->getExactMessage() : $constraint->getMaxMessage();
            throw new InvalidPropertyValueException(
                $this->generateMessage(self::INVALID_PROP_VALUE_TMPL, $constraintMessage),
                0,
                $this->getPropClassConcat(),
                $value
            );
        }

        if (null !== $constraint->getMin() && $length < $constraint->getMin()) {
            $exactlyOptionEnabled = $constraint->getMin() == $constraint->getMax();
            $constraintMessage = $exactlyOptionEnabled ? $constraint->getExactMessage() : $constraint->getMinMessage();
            throw new InvalidPropertyValueException(
                $this->generateMessage(self::INVALID_PROP_VALUE_TMPL, $constraintMessage),
                0,
                $this->getPropClassConcat(),
                $value
            );
        }
    }
}
