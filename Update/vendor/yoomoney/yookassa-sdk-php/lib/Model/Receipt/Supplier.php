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

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Class Supplier.
 *
 * Информация о поставщике товара или услуги.
 *
 * Можно передавать, если вы отправляете данные для формирования чека по сценарию [Сначала платеж, потом чек](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics#receipt-after-payment).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $name Наименование поставщика
 * @property string $phone Телефон пользователя. Указывается в формате ITU-T E.164
 * @property string $inn ИНН пользователя (10 или 12 цифр)
 */
class Supplier extends AbstractObject implements SupplierInterface
{
    /**
     * @var string|null Наименование поставщика
     */
    #[Assert\Type('string')]
    private ?string $_name = null;

    /**
     * @var string|null Телефон пользователя. Указывается в формате ITU-T E.164
     */
    #[Assert\Type('string')]
    #[Assert\Regex("/[0-9]{4,15}/")]
    private ?string $_phone = null;

    /**
     * @var string|null ИНН пользователя (10 или 12 цифр)
     */
    #[Assert\Type('string')]
    #[Assert\Regex(pattern: "/^\d{10}$|^\d{12}$/")]
    private ?string $_inn = null;

    public function getName(): ?string
    {
        return $this->_name;
    }

    /**
     * @param null|string $name Наименование поставщика
     *
     * @return self
     */
    public function setName(?string $name = null): self
    {
        $this->_name = $this->validatePropertyValue('_name', $name);
        return $this;
    }

    /**
     * Возвращает phone.
     *
     * @return string|null
     */
    public function getPhone(): ?string
    {
        return $this->_phone;
    }

    /**
     * @param null|string $phone Номер телефона пользователя в формате ITU-T E.164
     *
     * @return self
     */
    public function setPhone(?string $phone = null): self
    {
        $this->_phone = $this->validatePropertyValue('_phone', $phone);
        return $this;
    }

    public function getInn(): ?string
    {
        return $this->_inn;
    }

    /**
     * @param null|string $inn ИНН пользователя (10 или 12 цифр)
     *
     * @return self
     */
    public function setInn(?string $inn = null): self
    {
        $this->_inn = $this->validatePropertyValue('_inn', $inn);
        return $this;
    }
}
