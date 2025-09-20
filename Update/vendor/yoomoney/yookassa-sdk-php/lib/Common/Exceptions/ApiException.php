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

use Exception;
use YooKassa\Common\Errors\AbstractError;
use YooKassa\Common\Errors\ErrorFactory;

/**
 * Неожиданный код ошибки.
 */
class ApiException extends Exception
{
    /**
     * @var string|null
     */
    protected ?string $responseBody = '';

    /**
     * @var string[]
     */
    protected array $responseHeaders = [];

    /**
     * @var AbstractError|null Объект ошибки
     */
    protected ?AbstractError $error = null;

    /**
     * @var mixed|null Тип ошибки
     * @deprecated Устарело. Вместо него нужно использовать getError()->getType().
     */
    public mixed $type = null;

    /**
     * @var mixed|null Время в секундах до повторной попытки
     * @deprecated Устарело. Вместо него нужно использовать getError()->getRetryAfter().
     */
    public mixed $retryAfter = null;

    /**
     * Constructor.
     *
     * @param string $message Error message
     * @param int $code HTTP status code
     * @param string[] $responseHeaders HTTP header
     * @param string|null $responseBody HTTP body
     */
    public function __construct(string $message = '', int $code = 0, array $responseHeaders = [], ?string $responseBody = '')
    {
        parent::__construct($message, $code);
        $this->responseHeaders = $responseHeaders;
        $this->responseBody = $responseBody;
    }

    /**
     *
     * @return string[]
     */
    public function getResponseHeaders(): array
    {
        return $this->responseHeaders;
    }

    public function getResponseBody(): ?string
    {
        return $this->responseBody;
    }

    /**
     * Создает сообщение из объекта ошибки
     * @param mixed $responseBody
     * @return string
     */
    protected function createMessageFromError(mixed $responseBody): string
    {
        $errorData = json_decode($responseBody, true);
        $this->parseErrorBody($errorData ?? []);
        $message = '';

        if (!$this->getError()) {
            return $message;
        }

        if ($this->getError()->getType() !== null) {
            $this->type = $this->getError()->getType();
        }

        if ($this->getError()->getRetryAfter() !== null) {
            $this->retryAfter = $this->getError()->getRetryAfter();
        }

        if ($this->getError()->getDescription() !== null) {
            $message .= $this->getError()->getDescription() . '. ';
        }

        if ($this->getError()->getCode() !== null) {
            $message .= sprintf('Error code: %s. ', $this->getError()->getCode());
        }

        if ($this->getError()->getParameter() !== null) {
            $message .= sprintf('Parameter name: %s. ', $this->getError()->getParameter());
        }

        return $message;
    }

    /**
     * Подготавливает объект ошибки
     * @param array $errorData
     * @return void
     */
    protected function parseErrorBody(array $errorData): void
    {
        $this->error = (new ErrorFactory())->factoryFromArray($errorData);
    }

    /**
     * Возвращает объект ошибки
     * @return AbstractError|null
     */
    public function getError(): ?AbstractError
    {
        return $this->error;
    }

    /**
     * @deprecated Устарело. Вместо него нужно использовать getError()->getId()
     */
    public function getErrorId(): ?string
    {
        return $this->getError()?->getId();
    }
    /**
     * @deprecated Устарело. Вместо него нужно использовать getError()->getCode()
     */
    public function getErrorCode(): ?string
    {
        return $this->getError()?->getCode();
    }
    /**
     * @deprecated Устарело. Вместо него нужно использовать getError()->getDescription()
     */
    public function getErrorDescription(): ?string
    {
        return $this->getError()?->getDescription();
    }
    /**
     * @deprecated Устарело. Вместо него нужно использовать getError()->getParameter()
     */
    public function getErrorParameter(): ?string
    {
        return $this->getError()?->getParameter();
    }

}
