<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
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

namespace YooKassa\Common\Exceptions;

use RuntimeException;
use Throwable;
use YooKassa\Common\AbstractRequestInterface;

class InvalidRequestException extends RuntimeException
{
    private ?AbstractRequestInterface $errorRequest = null;

    /**
     * InvalidRequestException constructor.
     *
     * @param AbstractRequestInterface|string $error
     */
    public function __construct(mixed $error, int $code = 0, ?Throwable $previous = null)
    {
        if ($error instanceof AbstractRequestInterface) {
            $message = 'Failed to build request "' . $error::class . '": "' . $error->getLastValidationError() . '"';
            $this->errorRequest = $error;
        } else {
            $message = $error;
        }
        parent::__construct($message, $code, $previous);
    }

    public function getRequestObject(): ?AbstractRequestInterface
    {
        return $this->errorRequest;
    }
}
