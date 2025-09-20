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

namespace YooKassa\Common;

/**
 * Класс, представляющий модель ResponseObject.
 *
 * Класс HTTP ответа.
 *
 * @category Class
 * @package  YooKassa
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class ResponseObject
{
    /**
     * HTTP код ответа.
     *
     * @var int
     */
    protected int $code;

    /**
     * Массив заголовков ответа.
     */
    protected array $headers;

    /**
     * Тело ответа.
     *
     * @var mixed
     */
    protected mixed $body;

    public function __construct($config = null)
    {
        if (isset($config['headers'])) {
            $this->headers = $config['headers'];
        }

        if (isset($config['body'])) {
            $this->body = $config['body'];
        }

        if (isset($config['code'])) {
            $this->code = (int) $config['code'];
        }
    }

    /**
     * Возвращает массив заголовков ответа.
     *
     * @return array Массив заголовков ответа
     */
    public function getHeaders(): array
    {
        return $this->headers;
    }

    /**
     * Возвращает тело ответа.
     *
     * @return mixed Тело ответа
     */
    public function getBody(): mixed
    {
        return $this->body;
    }

    /**
     * Возвращает HTTP код ответа.
     *
     * @return int HTTP код ответа
     */
    public function getCode(): int
    {
        return $this->code;
    }
}
