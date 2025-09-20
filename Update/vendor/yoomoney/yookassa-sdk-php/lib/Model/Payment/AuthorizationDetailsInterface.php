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

namespace YooKassa\Model\Payment;

/**
 * Interface AuthorizationDetailsInterface - Данные об авторизации платежа.
 *
 * @property string $rrn Retrieval Reference Number — идентификатор банковской транзакции
 * @property string $authCode Код авторизации банковской карты
 * @property string $auth_code Код авторизации банковской карты
 * @property ThreeDSecure $threeDSecure Данные о прохождении пользователем аутентификации по 3‑D Secure
 * @property ThreeDSecure $three_d_secure Данные о прохождении пользователем аутентификации по 3‑D Secure
 */
interface AuthorizationDetailsInterface
{
    /**
     * Возвращает Retrieval Reference Number — идентификатор банковской транзакции.
     *
     * @return null|string Уникальный идентификатор транзакции
     */
    public function getRrn(): ?string;

    /**
     * Возвращает код авторизации банковской карты.
     *
     * @return null|string Код авторизации банковской карты
     */
    public function getAuthCode(): ?string;

    /**
     * Возвращает данные о прохождении пользователем аутентификации по 3‑D Secure.
     *
     * @return null|ThreeDSecure Объект с данными о прохождении пользователем аутентификации по 3‑D Secure
     */
    public function getThreeDSecure(): ?ThreeDSecure;
}
