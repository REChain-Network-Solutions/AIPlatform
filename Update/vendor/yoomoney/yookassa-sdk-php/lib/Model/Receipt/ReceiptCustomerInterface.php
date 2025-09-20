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

namespace YooKassa\Model\Receipt;

/**
 * Interface ReceiptCustomerInterface.
 *
 * @category Interface
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $fullName Для юрлица — название организации, для ИП и физического лица — ФИО.
 * @property string $full_name Для юрлица — название организации, для ИП и физического лица — ФИО.
 * @property string $phone Номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек.
 * @property string $email E-mail адрес плательщика на который будет выслан чек.
 * @property string $inn ИНН плательщика (10 или 12 цифр).
 */
interface ReceiptCustomerInterface
{
    /**
     * Возвращает название организации или ФИО физического лица.
     *
     * @return string|null Название организации или ФИО физического лица
     */
    public function getFullName(): ?string;

    /**
     * Возвращает номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек.
     *
     * @return string|null Номер телефона плательщика
     */
    public function getPhone(): ?string;

    /**
     * Возвращает адрес электронной почты на который будет выслан чек.
     *
     * @return string|null E-mail адрес плательщика
     */
    public function getEmail(): ?string;

    /**
     * Возвращает ИНН плательщика.
     *
     * @return string|null ИНН плательщика
     */
    public function getInn(): ?string;

    /**
     * Возвращает массив полей плательщика.
     */
    public function jsonSerialize(): array;
}
