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
use YooKassa\Validator\Exceptions\InvalidPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException;

/**
 * Информация о плательщике.
 *
 * @category Class
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
class ReceiptCustomer extends AbstractObject implements ReceiptCustomerInterface
{
    /**
     * @var string|null Для юрлица — название организации, для ИП и физического лица — ФИО
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 256)]
    private ?string $_full_name = null;

    /**
     * @var string|null Номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек.
     */
    #[Assert\Type('string')]
    #[Assert\Regex("/[0-9]{4,15}/")]
    private ?string $_phone = null;

    /**
     * @var string|null E-mail адрес плательщика на который будет выслан чек
     */
    #[Assert\Type('string')]
    #[Assert\Email]
    private ?string $_email = null;

    /**
     * @var string|null ИНН плательщика (10 или 12 цифр)
     */
    #[Assert\Type('string')]
    #[Assert\Regex(pattern: "/^\d{10}$|^\d{12}$/")]
    private ?string $_inn = null;

    /**
     * Возвращает для юрлица — название организации, для ИП и физического лица — ФИО.
     *
     * @return string|null Название организации или ФИО
     */
    public function getFullName(): ?string
    {
        return $this->_full_name;
    }

    /**
     * Устанавливает Название организации или ФИО.
     *
     * @param string|null $full_name Название организации или ФИО
     *
     * @return self
     */
    public function setFullName(?string $full_name = null): self
    {
        $this->_full_name = $this->validatePropertyValue('_full_name', $full_name);
        return $this;
    }

    /**
     * Возвращает номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек.
     *
     * @return string|null Номер телефона плательщика
     */
    public function getPhone(): ?string
    {
        return $this->_phone;
    }

    /**
     * Устанавливает номер телефона плательщика в формате ITU-T E.164 на который будет выслан чек.
     *
     * @param string|null $phone Номер телефона плательщика в формате ITU-T E.164
     *
     * @return self
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается, если в качестве значения была передана не строка
     */
    public function setPhone(?string $phone = null): self
    {
        $this->_phone = $this->validatePropertyValue('_phone', $phone);
        return $this;
    }

    /**
     * Возвращает адрес электронной почты на который будет выслан чек.
     *
     * @return string|null E-mail адрес плательщика
     */
    public function getEmail(): ?string
    {
        return $this->_email;
    }

    /**
     * Устанавливает адрес электронной почты на который будет выслан чек.
     *
     * @param string|null $email E-mail адрес плательщика
     *
     * @return self
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается, если в качестве значения была передана не строка
     * @throws InvalidPropertyValueException Выбрасывается если Email не соответствует формату
     */
    public function setEmail(?string $email = null): self
    {
        $this->_email = $this->validatePropertyValue('_email', $email);
        return $this;
    }

    /**
     * Возвращает ИНН плательщика.
     *
     * @return string|null
     */
    public function getInn(): ?string
    {
        return $this->_inn;
    }

    /**
     * Устанавливает ИНН плательщика.
     *
     * @param string|null $inn ИНН плательщика (10 или 12 цифр)
     *
     * @return self
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается, если в качестве значения была передана не строка
     * @throws InvalidPropertyValueException Выбрасывается если ИНН не соответствует формату 10 или 12 цифр
     */
    public function setInn(?string $inn = null): self
    {
        $this->_inn = $this->validatePropertyValue('_inn', $inn);
        return $this;
    }

    /**
     * Проверка на заполненность объекта.
     */
    public function isEmpty(): bool
    {
        $data = $this->getFullName() . $this->getEmail() . $this->getPhone() . $this->getInn();

        return empty($data);
    }

    public function jsonSerialize(): array
    {
        $result = [];

        $value = $this->getFullName();
        if (!empty($value)) {
            $result['full_name'] = $value;
        }
        $value = $this->getEmail();
        if (!empty($value)) {
            $result['email'] = $value;
        }
        $value = $this->getPhone();
        if (!empty($value)) {
            $result['phone'] = $value;
        }
        $value = $this->getInn();
        if (!empty($value)) {
            $result['inn'] = $value;
        }

        return $result;
    }
}
