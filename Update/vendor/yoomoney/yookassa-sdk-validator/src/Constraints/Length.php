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
#[Attribute(Attribute::TARGET_PROPERTY | Attribute::IS_REPEATABLE)]
class Length extends AbstractConstraint
{
    private string $maxMessage = 'This value is too long. It should have {{ limit }} character or less.';
    private string $minMessage = 'This value is too short. It should have {{ limit }} character or more.';
    private string $exactMessage = 'This value should have exactly {{ limit }} character.';
    private string $charsetMessage = 'This value does not match the expected {{ charset }} charset.';
    private ?string $max;
    private ?string $min;
    public string $charset = 'UTF-8';

    public function __construct(
        ?int $exactly = null,
        ?int $min = null,
        ?int $max = null,
        ?string $charset = null,
        ?string $exactMessage = null,
        ?string $minMessage = null,
        ?string $maxMessage = null,
    )
    {
        if (null !== $exactly && null === $min && null === $max) {
            $min = $max = $exactly;
        }

        $this->min = $min;
        $this->max = $max;
        $this->exactMessage = $exactMessage ?? $this->exactMessage;
        $this->minMessage = $minMessage ?? $this->minMessage;
        $this->maxMessage = $maxMessage ?? $this->maxMessage;
        $this->charset = $charset ?? $this->charset;
    }

    /**
     * @return string
     */
    public function getMaxMessage(): string
    {
        return str_replace('{{ limit }}', $this->max, $this->maxMessage);
    }

    /**
     * @return string
     */
    public function getMinMessage(): string
    {
        return str_replace('{{ limit }}', $this->min, $this->minMessage);
    }

    /**
     * @return string
     */
    public function getExactMessage(): string
    {
        return str_replace('{{ limit }}', $this->min, $this->exactMessage);
    }

    /**
     * @return string
     */
    public function getCharsetMessage(): string
    {
        return str_replace('{{ charset }}', $this->min, $this->charsetMessage);
    }

    /**
     * @return string|null
     */
    public function getMax(): ?string
    {
        return $this->max;
    }

    /**
     * @return string|null
     */
    public function getMin(): ?string
    {
        return $this->min;
    }

    /**
     * @return string
     */
    public function getCharset(): string
    {
        return $this->charset;
    }
}
