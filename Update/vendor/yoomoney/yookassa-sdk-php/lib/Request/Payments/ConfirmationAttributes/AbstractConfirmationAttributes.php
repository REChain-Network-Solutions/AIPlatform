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

namespace YooKassa\Request\Payments\ConfirmationAttributes;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\Payment\ConfirmationType;
use YooKassa\Request\Payments\Locale;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель AbstractConfirmationAttributes.
 *
 * Данные, необходимые для инициирования выбранного сценария подтверждения платежа пользователем. Подробнее о [сценариях подтверждения](/developers/payment-acceptance/getting-started/payment-process#user-confirmation)
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код сценария подтверждения
 * @property string $locale Язык интерфейса, писем и смс, которые будет видеть или получать пользователь
 */
abstract class AbstractConfirmationAttributes extends AbstractObject
{
    /**
     * @var string|null Код сценария подтверждения
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [ConfirmationType::class, 'getValidValues'])]
    protected ?string $_type = null;

    /**
     * @var string|null Язык интерфейса, писем и смс, которые будет видеть или получать пользователь
     */
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [Locale::class, 'getValidValues'])]
    protected ?string $_locale = null;

    /**
     * Возвращает код сценария подтверждения
     *
     * @return string|null Код сценария подтверждения
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает код сценария подтверждения.
     *
     * @param string|null $type Код сценария подтверждения
     *
     * @return self
     */
    public function setType(?string $type = null): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        return $this;
    }

    /**
     * Возвращает язык интерфейса, писем и смс
     *
     * @return string|null Язык интерфейса, писем и смс
     */
    public function getLocale(): ?string
    {
        return $this->_locale;
    }

    /**
     * Устанавливает язык интерфейса, писем и смс
     *
     * @param string|null $locale Язык интерфейса, писем и смс
     *
     * @return self
     */
    public function setLocale(?string $locale = null): self
    {
        $this->_locale = $this->validatePropertyValue('_locale', $locale);
        return $this;
    }
}
