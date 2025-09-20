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
 * Interface SupplierInterface.
 *
 * Информация о поставщике товара или услуги.
 *
 * Можно передавать, если вы отправляете данные для формирования чека по сценарию - сначала платеж, потом чек.
 *
 * @category Interface
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $name Наименование поставщика
 * @property string $phone Телефон пользователя. Указывается в формате ITU-T E.164
 * @property string $inn ИНН пользователя (10 или 12 цифр)
 */
interface SupplierInterface
{
    /**
     * Возвращает наименование поставщика.
     */
    public function getName(): ?string;

    /**
     * Устанавливает наименование поставщика.
     *
     * @param null|string $name Наименование поставщика
     */
    public function setName(?string $name);

    /**
     * Возвращает Телефон пользователя. Указывается в формате ITU-T E.164.
     *
     * @return null|string Телефон пользователя
     */
    public function getPhone(): ?string;

    /**
     * Устанавливает Телефон пользователя. Указывается в формате ITU-T E.164.
     *
     * @param null|string $phone Телефон пользователя
     */
    public function setPhone(?string $phone);

    /**
     * Возвращает ИНН пользователя (10 или 12 цифр).
     *
     * @return null|string ИНН пользователя
     */
    public function getInn(): ?string;

    /**
     * Устанавливает ИНН пользователя (10 или 12 цифр).
     *
     * @param null|string $inn ИНН пользователя
     */
    public function setInn(?string $inn);
}
