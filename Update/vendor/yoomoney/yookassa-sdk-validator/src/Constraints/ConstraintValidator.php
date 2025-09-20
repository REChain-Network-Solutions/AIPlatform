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

abstract class ConstraintValidator
{
    protected const INVALID_PROP_VALUE_TMPL = 'Invalid {{ value }} value in {{ class }}';
    protected const INVALID_PROP_VALUE_TYPE_TMPL = 'Invalid {{ value }} value type in {{ class }}';
    protected const EMPTY_PROP_VALUE_TMPL = 'Empty {{ value }} value in {{ class }}';

    protected string $className;
    protected string $propertyName;

    public function __construct(string $className, string $propertyName)
    {
        $this->className = $className;
        $this->propertyName = $propertyName;
    }

    protected function generateMessage(string $message, ?string $extraMessage = null): string
    {
        $message = str_replace(['{{ class }}', '{{ value }}'], [$this->className, $this->propertyName], $message);
        $message .= $extraMessage ? ('. ' . $extraMessage) : '';

        return $message;
    }

    protected function getPropClassConcat(): string
    {
        return $this->className . '.' . $this->propertyName;
    }
}
