# AlbumentationsX License Guide

AlbumentationsX uses a sustainable dual licensing model. This guide explains the licensing model and what it means for you.

## Quick Summary

- **AlbumentationsX** is dual-licensed: AGPL-3.0 (open source) / Commercial
- **Open source projects** can use AlbumentationsX for free under AGPL
- **Commercial projects** need a commercial license for proprietary use

## Understanding the License Change

### Background

AlbumentationsX uses a dual license model. This approach ensures sustainable maintenance of a widely-used open source project:

- The project has grown to serve thousands of companies and researchers worldwide
- Maintenance, bug fixes, and feature development require significant time investment
- The original team members have moved on to new opportunities
- Financial support through donations has been minimal (covering ~2.5% of maintainer living costs)
- Companies have established processes for purchasing licenses but not for donations

### The Dual License Model

AlbumentationsX offers two licensing options:

1. **AGPL-3.0 License**: For open source projects
   - Free to use in open source projects
   - **IMPORTANT**: Your entire project must be licensed under AGPL-3.0
   - You CANNOT use AlbumentationsX in MIT, Apache, BSD, or other permissively licensed projects
   - Requires sharing source code of your entire application
   - If you provide a network service, you must provide source code to users

2. **Commercial License**: For proprietary/commercial use
   - Use AlbumentationsX in closed-source applications
   - Use AlbumentationsX in projects with any license (MIT, Apache, proprietary, etc.)
   - No requirement to share your source code
   - Professional support options available
   - [View pricing](https://albumentations.ai/pricing)

### Why AGPL?

AGPL (Affero General Public License) extends GPL to network services. Key points:

- **Copyleft**: If you use AGPL software, your entire project must also be AGPL
- Cannot be combined with permissive licenses (MIT, Apache, BSD) without converting the entire project to AGPL
- If you use AGPL software in a network service, you must provide source code to users
- Ensures improvements benefit the entire community
- Encourages contribution back to the open source ecosystem
- Compatible only with other AGPL/GPL licensed software

## Your Options

### For Open Source Projects

**Important**: You can only use AlbumentationsX under AGPL if your entire project is also AGPL-licensed.

Continue using AlbumentationsX under AGPL:

```bash
pip install albumentationsx
```

Your code stays the same:
```python
import albumentations as A

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
```

**Requirements**:
- Your project must be licensed under AGPL-3.0
- You must provide source code when distributing your software
- If you run a network service, you must provide source code to users
- You cannot use AlbumentationsX in MIT, Apache, BSD, or other permissively licensed projects

### For Commercial/Proprietary Projects

You have two paths:

1. **Purchase a Commercial License**
   - Continue using AlbumentationsX in proprietary software
   - No source code disclosure required
   - Support the project's continued development
   - [Get commercial license](https://albumentations.ai/pricing)

2. **Open Source Your Code Under AGPL**
   - Convert your entire project to AGPL-3.0 license
   - This means abandoning MIT, Apache, BSD, or other permissive licenses
   - Must provide source code when distributing
   - Valid approach if AGPL aligns with your project goals

## Benefits of the Dual License Model

### For the Community

- **Sustainable Development**: Full-time maintainer focus on the project
- **Faster Innovation**: More resources for new features and optimizations
- **Better Support**: Professional bug fixing and feature development
- **Long-term Viability**: Ensures the project's future

### For Commercial Users

- **Legal Clarity**: Clear licensing terms for commercial use
- **Professional Support**: Access to dedicated support channels
- **Compliance**: Easier to justify to legal departments
- **Investment Protection**: Ensures continued development of a critical dependency

### Success Stories

Projects like [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) have successfully used dual licensing to achieve:
- Full-time development teams
- Rapid feature delivery
- Professional support
- Growing open source community

## FAQ

### Q: Do I need to pay to use AlbumentationsX?

**A:** Only if you're using it in proprietary/commercial software OR if your open source project uses a non-AGPL license (like MIT, Apache, BSD). Open source projects licensed under AGPL can use it for free.

### Q: What if I'm a student or researcher?

**A:** If you're publishing your code under AGPL, you can use AlbumentationsX for free. If you want to use a different license (like MIT) or keep your code private, you'll need a commercial license.

### Q: Can I use AlbumentationsX in my MIT/Apache/BSD licensed project?

**A:** No, not under the AGPL license. AGPL is a copyleft license that requires your entire project to be AGPL. To use AlbumentationsX in projects with permissive licenses, you need a commercial license.

### Q: Can I try AlbumentationsX before purchasing a commercial license?

**A:** Yes! You can evaluate AlbumentationsX in development. A commercial license is only required for production deployment in proprietary systems.

### Q: What about existing projects?

**A:** To use AlbumentationsX, you need to comply with the licensing terms - either use AGPL for your entire project or purchase a commercial license.

### Q: How do I know if I need a commercial license?

**A:** You need a commercial license if your software is:
- Closed source / proprietary
- Open source with a non-AGPL license (MIT, Apache, BSD, ISC, etc.)
- Used internally in a company without public source release
- Distributed as a commercial product without source code
- Part of a SaaS/cloud service where you don't want to share source code

The only case where you DON'T need a commercial license is if your entire project is licensed under AGPL-3.0 and you comply with all AGPL requirements.

### Q: What if I contribute to AlbumentationsX?

**A:** Contributors grant rights to use their contributions under both AGPL and commercial licenses, enabling the dual licensing model.


## Support

- **Open Source Users**: [GitHub Issues](https://github.com/albumentations-team/AlbumentationsX/issues)
- **Commercial Users**: Dedicated support channels (included with license)
- **Community**: [Discord](https://discord.gg/AKPrrDYNAt)

## Conclusion

The transition to dual licensing ensures AlbumentationsX can continue to grow and improve while serving both open source and commercial communities. This model has proven successful for many projects and provides the resources needed for professional maintenance and development.

For commercial licensing inquiries, visit [our pricing page](https://albumentations.ai/pricing).
