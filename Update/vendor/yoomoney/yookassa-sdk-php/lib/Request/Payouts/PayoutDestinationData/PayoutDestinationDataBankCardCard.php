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

namespace YooKassa\Request\Payouts\PayoutDestinationData;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PayoutDestinationDataBankCardCard.
 *
 * Данные банковской карты для выплаты.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $number Номер банковской карты. Формат: только цифры, без пробелов.
 */
class PayoutDestinationDataBankCardCard extends AbstractObject
{
    /**
     * Номер банковской карты. Формат: только цифры, без пробелов. Пример: ~`5555555555554477`
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex("/[0-9]{16,19}/")]
    private ?string $_number = null;

    /**
     * Возвращает последние 4 цифры номера карты.
     *
     * @return string|null Последние 4 цифры номера карты
     */
    public function getNumber(): ?string
    {
        return $this->_number;
    }

    /**
     * Устанавливает номер банковской карты.
     *
     * @param string|null $number Номер банковской карты. Формат: только цифры, без пробелов. Пример: ~`5555555555554477`
     *
     * @return self
     */
    public function setNumber(?string $number = null): self
    {
        $this->_number = $this->validatePropertyValue('_number', $number);
        return $this;
    }
}
