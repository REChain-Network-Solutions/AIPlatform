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

use Attribute;

/**
 * @Annotation
 * @Target({"PROPERTY", "ANNOTATION"})
 */
#[Attribute(Attribute::TARGET_PROPERTY)]
class Choice extends AbstractConstraint
{
    private string $message = 'The value you selected is not a valid choice.';
    private string $multipleMessage = 'One or more of the given values is invalid.';
    private ?array $choices;
    private $callback;
    private bool $multiple = false;
    private bool $match = true;

    public function __construct(
        ?array $choices = null,
        callable|string|null $callback = null,
        ?bool $multiple = null,
        ?string $message = null,
        ?string $multipleMessage = null,
        ?bool $match = null,
    ) {
        $this->choices = $choices;
        $this->callback = $callback ?? $this->callback;
        $this->multiple = $multiple ?? $this->multiple;
        $this->message = $message ?? $this->message;
        $this->multipleMessage = $multipleMessage ?? $this->multipleMessage;
        $this->match = $match ?? $this->match;
    }

    /**
     * @return string
     */
    public function getMessage(): string
    {
        return $this->message;
    }

    /**
     * @return array|null
     */
    public function getChoices(): ?array
    {
        return $this->choices;
    }

    /**
     * @return bool
     */
    public function isMultiple(): bool
    {
        return $this->multiple;
    }

    /**
     * @return bool
     */
    public function isMatch(): bool
    {
        return $this->match;
    }

    /**
     * @return string
     */
    public function getMultipleMessage(): string
    {
        return $this->multipleMessage;
    }

    /**
     * @return callable|string|null
     */
    public function getCallback(): callable|string|null
    {
        return $this->callback;
    }
}
