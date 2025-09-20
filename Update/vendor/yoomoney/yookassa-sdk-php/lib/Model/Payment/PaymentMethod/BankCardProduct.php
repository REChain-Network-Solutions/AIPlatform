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

namespace YooKassa\Model\Payment\PaymentMethod;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель BankCardProduct.
 *
 * Карточный продукт платежной системы, с которым ассоциирована банковская карта.
 * Например, карточные продукты платежной системы Мир: `Mir Classic`, `Mir Classic Credit`, `MIR Privilege Plus` и другие.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $code Код карточного продукта. Пример: ~`MCP`
 * @property string|null $name Название карточного продукта. Пример: ~`MIR Privilege`
*/
class BankCardProduct extends AbstractObject
{
    /**
     * Код карточного продукта. Пример: ~`MCP`
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_code = null;

    /**
     * Название карточного продукта. Пример: ~`MIR Privilege`
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_name = null;

    /**
     * Возвращает code.
     *
     * @return string|null
     */
    public function getCode(): ?string
    {
        return $this->_code;
    }

    /**
     * Устанавливает code.
     *
     * @param string|null $code Код карточного продукта. Пример: ~`MCP`
     *
     * @return self
     */
    public function setCode(?string $code = null): self
    {
        $this->_code = $this->validatePropertyValue('_code', $code);
        return $this;
    }

    /**
     * Возвращает name.
     *
     * @return string|null
     */
    public function getName(): ?string
    {
        return $this->_name;
    }

    /**
     * Устанавливает name.
     *
     * @param string|null $name Название карточного продукта. Пример: ~`MIR Privilege`
     *
     * @return self
     */
    public function setName(?string $name = null): self
    {
        $this->_name = $this->validatePropertyValue('_name', $name);
        return $this;
    }

}

