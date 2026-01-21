import { DNA } from 'react-loader-spinner'

function Spinner({ size = 40, visible = true, ariaLabel = 'dna-loading', wrapperClass = 'dna-wrapper', dnaColorOne = '#FF0000', dnaColorTwo = '#00ff00' }) {
  return (
    <DNA
      visible={visible}
      height={size}
      width={size}
      dnaColorOne={dnaColorOne}
      dnaColorTwo={dnaColorTwo}
      ariaLabel={ariaLabel}
      wrapperStyle={{ display: 'inline-block' }}
      wrapperClass={wrapperClass}
    />
  )
}

export default Spinner

/*
<Oval
      height={size}
      width={size}
      color={color}
      wrapperStyle={{ display: 'inline-block' }}
      wrapperClass=""
      visible={visible}
      ariaLabel={ariaLabel}
      secondaryColor={color}
      strokeWidth={2}
      strokeWidthSecondary={2}
    />
*/