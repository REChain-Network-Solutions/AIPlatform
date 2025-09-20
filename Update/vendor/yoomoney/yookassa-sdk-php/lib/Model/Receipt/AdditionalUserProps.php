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
 * Class AdditionalUserProps.
 *
 * Дополнительный реквизит пользователя (тег в 54 ФЗ — 1084). <br/>Можно передавать, если вы отправляете данные для формирования чека по сценарию [Сначала платеж, потом чек](/developers/payment-acceptance/receipts/54fz/other-services/basics#receipt-after-payment)
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $name Наименование дополнительного реквизита пользователя (тег в 54 ФЗ — 1085). Не более 64 символов.
 * @property string $value Значение дополнительного реквизита пользователя (тег в 54 ФЗ — 1086). Не более 234 символов.
 */
class AdditionalUserProps extends AbstractObject
{
    /** @var int Максимальная длинна наименования реквизита */
    public const NAME_MAX_LENGTH = 64;

    /** @var int Максимальная длинна значение наименования реквизита */
    public const VALUE_MAX_LENGTH = 234;

    /**
     * Наименование дополнительного реквизита пользователя (тег в 54 ФЗ — 1085). Не более 64 символов.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::NAME_MAX_LENGTH)]
    private ?string $_name = null;

    /**
     * Значение дополнительного реквизита пользователя (тег в 54 ФЗ — 1086). Не более 234 символов.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::VALUE_MAX_LENGTH)]
    private ?string $_value = null;

    /**
     * Возвращает наименование дополнительного реквизита пользователя.
     *
     * @return string|null Наименование дополнительного реквизита пользователя
     */
    public function getName(): ?string
    {
        return $this->_name;
    }

    /**
     * Устанавливает наименование дополнительного реквизита пользователя.
     *
     * @param string|null $name Наименование дополнительного реквизита пользователя
     *
     * @return self
     */
    public function setName(?string $name = null): self
    {
        $this->_name = $this->validatePropertyValue('_name', $name);
        return $this;
    }

    /**
     * Возвращает значение дополнительного реквизита пользователя.
     *
     * @return string|null Значение дополнительного реквизита пользователя
     */
    public function getValue(): ?string
    {
        return $this->_value;
    }

    /**
     * Устанавливает значение дополнительного реквизита пользователя.
     *
     * @param string|null $value Значение дополнительного реквизита пользователя
     *
     * @return self
     */
    public function setValue(?string $value = null): self
    {
        $this->_value = $this->validatePropertyValue('_value', $value);
        return $this;
    }
}
