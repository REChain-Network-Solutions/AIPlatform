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

namespace YooKassa\Common\Errors;

use YooKassa\Common\AbstractEnum;

/**
 * Класс, представляющий модель ErrorCode.
 *
 * Код ошибки.
 *
 * Возможные значения:
 * - `error` — Общая ошибка.
 * - `invalid_request` — Запрос не может быть обработан.
 * - `invalid_credentials` — В заголовке Authorization указан неверный ключ.
 * - `forbidden` — Секретный ключ указан верно, но не хватает прав для совершения операции.
 * - `not_found` — Сущность не найдена.
 * - `gone` — Сущность была раньше, но была умышленно удалена и теперь недоступна.
 * - `too_many_requests` — Слишком много запросов одновременно отправляется в API.
 * - `internal_server_error` — Внутренняя ошибка сервера ЮKassa.
 * - `unknown` — Для неописанных типов ошибок.
 *
 * Подробнее об [ошибках](https://yookassa.ru/developers/using-api/response-handling/response-format#error)
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class ErrorCode extends AbstractEnum
{
    /** Общая ошибка */
    public const ERROR = 'error';
    /** Запрос не может быть обработан. Причиной может быть неправильный синтаксис запроса, ошибка в обязательных параметрах запроса, их отсутствие или неподдерживаемый метод. */
    public const INVALID_REQUEST = 'invalid_request';
    /** В заголовке Authorization указан неверный ключ. */
    public const INVALID_CREDENTIALS = 'invalid_credentials';
    /** Секретный ключ указан верно, но не хватает прав для совершения операции. */
    public const FORBIDDEN = 'forbidden';
    /** Сущность не найдена. */
    public const NOT_FOUND = 'not_found';
    /** Сущность была раньше, но была умышленно удалена и теперь недоступна. */
    public const GONE = 'gone';
    /** Слишком много запросов одновременно отправляется в API. Повторите запрос позже. */
    public const TOO_MANY_REQUESTS = 'too_many_requests';
    /** Внутренняя ошибка сервера ЮKassa. */
    public const INTERNAL_SERVER_ERROR = 'internal_server_error';
    /** Для неописанных кодов ошибок. */
    public const UNKNOWN = 'unknown';

    /**
     * Возвращает список доступных значений
     * @return string[]
     */
    protected static array $validValues = [
        self::ERROR => true,
        self::INVALID_REQUEST => true,
        self::INVALID_CREDENTIALS => true,
        self::FORBIDDEN => true,
        self::NOT_FOUND => true,
        self::GONE => true,
        self::TOO_MANY_REQUESTS => true,
        self::INTERNAL_SERVER_ERROR => true,
        self::UNKNOWN => false,
    ];
}
