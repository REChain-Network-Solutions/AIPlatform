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

namespace YooKassa\Client;

use Psr\Log\LoggerInterface;
use YooKassa\Common\ResponseObject;

/**
 * Interface ApiClientInterface.
 *
 * @category Interface
 * @package  YooKassa
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
interface ApiClientInterface
{
    /**
     * Создает CURL запрос, получает и возвращает обработанный ответ
     *
     * @param string $path URL запроса
     * @param string $method HTTP метод
     * @param array $queryParams Массив GET параметров запроса
     * @param null|string $httpBody Тело запроса
     * @param array $headers Массив заголовков запроса
     */
    public function call(string $path, string $method, array $queryParams, ?string $httpBody = null, array $headers = []): ResponseObject;

    /**
     * Устанавливает объект для логирования.
     *
     * @param null|LoggerInterface $logger Объект для логирования
     */
    public function setLogger(?LoggerInterface $logger);

    /**
     * Возвращает UserAgent.
     */
    public function getUserAgent(): UserAgent;

    /**
     * Устанавливает shopId магазина.
     *
     * @param int|string|null $shopId shopId магазина
     */
    public function setShopId(mixed $shopId): mixed;

    /**
     * Устанавливает секретный ключ магазина.
     */
    public function setShopPassword(?string $shopPassword): mixed;

    /**
     * Устанавливает OAuth-токен магазина.
     */
    public function setBearerToken(?string $bearerToken): mixed;

    /**
     * Устанавливает настройки.
     */
    public function setConfig(array $config);

    /**
     * Устанавливает дополнительные настройки curl.
     */
    public function setAdvancedCurlOptions(): void;
}
